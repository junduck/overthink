"""SIGRegTokenizer: Autoencoder with SIGReg-regulated quantization.

Replaces KronosTokenizer's heuristic stack (per-window z-score + causal
attention encoder + L2 norm + Binary Spherical Quantization with entropy/
commitment regularization) with a principled mechanism:

    SIGReg forces encoder output z ~ N(0, I)
    => sign(z) gives uniform binary codes by construction
    => no entropy regularization, no commitment loss, no input normalization

The encoder uses causal attention within a fixed-size sliding window.
Each bar's token depends on the previous W-1 bars in its window — not on
absolute position or total sequence length. Same W-bar pattern always
produces the same token.

Architecture:
    raw_ohlcv [B, T, d_in]
        -> Linear embed (d_in -> d_model)
        -> Chunked causal TransStack encoder (window_size W)
        -> Linear quant_embed (d_model -> codebook_dim)
        -> SIGReg loss on pre-quantization latents
        -> sign() quantization (straight-through estimator)
        -> split into s1 (coarse) and full codebook
        -> s1 path: Linear -> TransStack decoder -> Linear head -> recon_s1
        -> full path: Linear -> TransStack decoder -> Linear head -> recon_full
"""

import torch
import torch.nn.functional as F
from torch import nn

from overthink.block import TransStack
from overthink.layer import SIGReg
from overthink.layer.linear import Linear
from overthink.layer.utils import get_torch_dtype

from .sigreg_tokenizer_config import SIGRegTokenizerConfig


class SIGRegTokenizer(nn.Module):
    """Autoencoder with SIGReg-regulated binary quantization and windowed causal encoder.

    The encoder chunks the input into windows of size W, applies causal attention
    within each window. Bar at position t within its window sees the previous W-1
    bars. Same local pattern always produces the same token regardless of absolute
    position or total sequence length.

    Args:
        config: SIGRegTokenizerConfig with architecture and SIGReg params.
    """

    def __init__(self, config: SIGRegTokenizerConfig):
        super().__init__()
        self.config = config
        self.d_in = config.d_in
        self.d_model = config.d_model
        self.s1_bits = config.s1_bits
        self.s2_bits = config.s2_bits
        self.codebook_dim = config.codebook_dim
        self.window_size = config.encoder_window
        dtype = get_torch_dtype(config.model_dtype)

        self.embed = Linear(self.d_in, self.d_model, dtype=dtype)
        self.head = Linear(self.d_model, self.d_in, dtype=dtype)

        self.encoder = TransStack(
            layer_num=config.n_enc_layers,
            hidden_size=config.d_model,
            head_num=config.n_heads,
            query_grp=0,
            dropout=config.ffn_dropout_p,
            causal=True,
            expansion_factor=config.ff_dim / config.d_model,
            eps=1e-5,
            dtype=dtype,
        )

        self.decoder = TransStack(
            layer_num=config.n_dec_layers,
            hidden_size=config.d_model,
            head_num=config.n_heads,
            query_grp=0,
            dropout=config.ffn_dropout_p,
            causal=False,
            expansion_factor=config.ff_dim / config.d_model,
            eps=1e-5,
            dtype=dtype,
        )

        self.quant_embed = Linear(self.d_model, self.codebook_dim, dtype=dtype)
        self.post_quant_embed_s1 = Linear(self.s1_bits, self.d_model, dtype=dtype)
        self.post_quant_embed = Linear(self.codebook_dim, self.d_model, dtype=dtype)

        self.sigreg = SIGReg(
            embed_dim=self.codebook_dim,
            num_slices=config.sigreg_num_slices,
            t_max=config.sigreg_t_max,
            n_points=config.sigreg_n_points,
        )

    @staticmethod
    def quantize(z: torch.Tensor) -> torch.Tensor:
        """Binary sign quantization with straight-through estimator."""
        zhat = torch.sign(z)
        return z + (zhat - z).detach()

    def bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert bipolar codes to integer indices."""
        binary = (bits >= 0).long()
        mask = 2 ** torch.arange(
            bits.shape[-1], device=bits.device, dtype=torch.long
        )
        return (binary * mask).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Convert integer indices back to bipolar codes."""
        mask = 2 ** torch.arange(n_bits - 1, -1, -1, device=indices.device, dtype=torch.long)
        binary = (indices.unsqueeze(-1) & mask) != 0
        return binary.float() * 2 - 1

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input to pre-quantization latents with sliding window.

        Chunks the sequence into windows of size W with stride W/2 (overlapping).
        Each chunk is encoded independently with causal attention. For overlapping
        positions, we take the output where the position had the most context
        (i.e., closer to the end of its chunk).

        Args:
            x: [B, T, d_in] raw OHLCV.

        Returns:
            [B, T, codebook_dim] pre-quantization latents.
        """
        z = self.embed(x)
        B, T, D = z.shape
        W = self.window_size
        S = W // 2  # stride

        if T <= W:
            z = self.encoder(z)
            return self.quant_embed(z)

        # Pad right to cover all positions
        n_chunks = (T - W + S - 1) // S + 1
        total_len = W + (n_chunks - 1) * S
        pad = total_len - T
        if pad > 0:
            z = F.pad(z, (0, 0, 0, pad))

        result = z.new_zeros(B, total_len, D)
        count = z.new_zeros(B, total_len, 1)

        for i in range(n_chunks):
            start = i * S
            chunk = z[:, start:start + W, :]
            encoded = self.encoder(chunk)

            result[:, start:start + W, :] += encoded
            count[:, start:start + W, :] += 1

        result = result / count.clamp(min=1)
        return self.quant_embed(result[:, :T, :])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, quantize, reconstruct."""
        z = self._encode_latent(x)
        sigreg_loss = self.sigreg(z)
        quantized = self.quantize(z)

        quantized_s1 = quantized[:, :, : self.s1_bits]
        z_s1 = self.post_quant_embed_s1(quantized_s1)
        z_full = self.post_quant_embed(quantized)

        z_s1 = self.decoder(z_s1)
        z_full = self.decoder(z_full)

        recon_s1 = self.head(z_s1)
        recon_full = self.head(z_full)

        z_indices = self.bits_to_indices(quantized)
        return (recon_s1, recon_full), sigreg_loss, quantized, z_indices

    @torch.no_grad()
    def encode(
        self, x: torch.Tensor, half: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode raw input to discrete token indices."""
        z = self._encode_latent(x)
        quantized = self.quantize(z)

        if half:
            q_s1 = quantized[:, :, : self.s1_bits]
            q_s2 = quantized[:, :, self.s1_bits :]
            return self.bits_to_indices(q_s1), self.bits_to_indices(q_s2)
        return self.bits_to_indices(quantized)

    def decode(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], half: bool = False
    ) -> torch.Tensor:
        """Decode token indices back to OHLCV space."""
        if half:
            s1_ids, s2_ids = x
            codes_s1 = self.indices_to_codes(s1_ids, self.s1_bits)
            codes_s2 = self.indices_to_codes(s2_ids, self.s2_bits)
            quantized = torch.cat([codes_s1, codes_s2], dim=-1)
        else:
            quantized = self.indices_to_codes(x, self.codebook_dim)

        z = self.post_quant_embed(quantized)
        z = self.decoder(z)
        return self.head(z)
