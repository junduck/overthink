"""OverthinkBSQ: Overthink's recursive reasoning on discrete BSQ tokens.

Like an LLM: discrete token input → reasoning loop → next-token classification.

Architecture:
  (s1, s2) ids
      │
      ▼  HierarchicalTokenEmbedding
  [B, S, H]
      │
      ▼  ReasoningBlock: stateful local/global loop, one shared TransStack
  [B, S, H]
      │
      ├─► head_s1: Linear(H, vocab_size)  → CE loss
      └─► head_s2: Linear(H, vocab_size)  → CE loss
"""

import math

import torch
from torch import nn

from overthink.block import ReasoningBlock
from overthink.layer import Linear, RoPE, SwiGLU, TemporalEmbedding
from overthink.layer.utils import get_torch_dtype

from .bsq_config import BSQConfig


class HierarchicalTokenEmbedding(nn.Module):
    """Embed (s1_id, s2_id) token pairs into hidden dimension.

    Two independent embedding tables fused via linear + SwiGLU.

    Args:
        vocab_size: Size of each sub-vocabulary (1024 for 10-bit)
        hidden_size: Model hidden dimension
        dtype: Parameter data type
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.emb_s1 = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.emb_s2 = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.fusion = Linear(hidden_size * 2, hidden_size, bias=False, dtype=dtype)
        self.mixer = SwiGLU(hidden_size, expansion_factor=1.0, dtype=dtype)
        self.scale = math.sqrt(hidden_size)

        nn.init.normal_(self.emb_s1.weight, mean=0, std=hidden_size**-0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=hidden_size**-0.5)

    def forward(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor) -> torch.Tensor:
        s1_emb = self.emb_s1(s1_ids) * self.scale
        s2_emb = self.emb_s2(s2_ids) * self.scale
        x = self.fusion(torch.cat([s1_emb, s2_emb], dim=-1))
        return self.mixer(x)


class OverthinkBSQ(nn.Module):
    """Overthink with discrete BSQ token I/O.

    Uses ReasoningBlock for stateful hierarchical reasoning with persistent
    local and global states. One shared TransStack, looped to strengthen
    signals through recursive mixing.
    """

    def __init__(self, config: BSQConfig):
        super().__init__()
        self.config = config
        dtype = get_torch_dtype(config.model_dtype)

        rope = (
            RoPE(
                dim=config.hidden_size // config.head_num,
                max_seq_len=config.rope_max_seq_len,
                theta=config.rope_theta,
                dtype=dtype,
            )
            if config.use_rope
            else None
        )

        self.token_embed = HierarchicalTokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            dtype=dtype,
        )

        if config.use_temporal:
            self.temporal_embed = TemporalEmbedding(
                hidden_size=config.hidden_size,
                embed_dim=config.temporal_embed_dim,
                dtype=dtype,
            )
        else:
            self.temporal_embed = None

        self.reasoning = ReasoningBlock(
            hidden_size=config.hidden_size,
            stack_depth=config.stack_depth,
            local_steps=config.local_steps,
            global_steps=config.global_steps,
            head_num=config.head_num,
            query_grp=config.query_group,
            dropout=config.attn_dropout,
            causal=config.use_causal,
            expansion_factor=config.expansion_factor,
            eps=config.rms_eps,
            rope=rope,
            dtype=dtype,
        )

        self.head_s1 = Linear(
            config.hidden_size, config.vocab_size, bias=True, dtype=dtype
        )
        self.head_s2 = Linear(
            config.hidden_size, config.vocab_size, bias=True, dtype=dtype
        )

        with torch.no_grad():
            self.head_s1.w.mul_(0.01)
            if self.head_s1.b is not None:
                self.head_s1.b.zero_()
            self.head_s2.w.mul_(0.01)
            if self.head_s2.b is not None:
                self.head_s2.b.zero_()

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: embed tokens, reason, return logits at ALL positions.

        Args:
            s1_ids: [B, S] coarse subtoken IDs
            s2_ids: [B, S] fine subtoken IDs
            timestamps: [B, S, 5] temporal features (optional)

        Returns:
            (s1_logits, s2_logits) each [B, S, vocab_size]
        """
        x = self.token_embed(s1_ids, s2_ids)

        if self.temporal_embed is not None and timestamps is not None:
            x = x + self.temporal_embed(timestamps)

        state = self.reasoning(x)

        s1_logits = self.head_s1(state)
        s2_logits = self.head_s2(state)
        return s1_logits, s2_logits

    def predict_next(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next token: returns logits for the last position.

        Args:
            s1_ids: [B, S] coarse subtoken IDs
            s2_ids: [B, S] fine subtoken IDs
            timestamps: [B, S, 5] temporal features (optional)

        Returns:
            (s1_logits, s2_logits) each [B, vocab_size]
        """
        s1_logits, s2_logits = self.forward(s1_ids, s2_ids, timestamps)
        return s1_logits[:, -1, :], s2_logits[:, -1, :]

    def compute_loss(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute next-token prediction loss (shifted CE).

        Args:
            s1_ids: [B, S] full token sequence
            s2_ids: [B, S] full token sequence
            timestamps: [B, S, 5] temporal features (optional)

        Returns:
            (total_loss, s1_loss, s2_loss)
        """
        V = self.config.vocab_size
        ts = timestamps[:, :-1] if timestamps is not None else None
        s1_logits, s2_logits = self.forward(s1_ids[:, :-1], s2_ids[:, :-1], ts)

        s1_logits = s1_logits.reshape(-1, V)
        s2_logits = s2_logits.reshape(-1, V)
        s1_target = s1_ids[:, 1:].reshape(-1)
        s2_target = s2_ids[:, 1:].reshape(-1)

        loss_s1 = nn.functional.cross_entropy(s1_logits, s1_target)
        loss_s2 = nn.functional.cross_entropy(s2_logits, s2_target)
        loss = loss_s1 + loss_s2
        return loss, loss_s1, loss_s2

    def train_step(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        timestamps: torch.Tensor | None = None,
    ) -> tuple[float, float, float]:
        """Single training step: forward + shifted CE loss + backward.

        Args:
            s1_ids: [B, S] token sequences (S must be >= 2)
            s2_ids: [B, S] token sequences
            optimizer: Optimizer
            timestamps: [B, S, 5] temporal features (optional)

        Returns:
            (total_loss, s1_loss, s2_loss)
        """
        self.train()
        optimizer.zero_grad()

        loss, loss_s1, loss_s2 = self.compute_loss(s1_ids, s2_ids, timestamps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item(), loss_s1.item(), loss_s2.item()

    @torch.no_grad()
    def autoregressive_generate(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        horizon: int,
        timestamps: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate future tokens autoregressively with top-p sampling.

        Args:
            s1_ids: [B, S] initial context s1 IDs
            s2_ids: [B, S] initial context s2 IDs
            horizon: Number of steps to generate
            timestamps: [B, S, 5] temporal features (optional)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            (gen_s1, gen_s2) each [B, horizon] generated token IDs
        """
        self.eval()
        ctx_s1 = s1_ids
        ctx_s2 = s2_ids
        ctx_ts = timestamps

        gen_s1_list: list[torch.Tensor] = []
        gen_s2_list: list[torch.Tensor] = []

        for _ in range(horizon):
            s1_logits, s2_logits = self.predict_next(ctx_s1, ctx_s2, None)

            s1_logits = s1_logits / temperature
            s2_logits = s2_logits / temperature

            next_s1 = _sample_top_p(s1_logits, top_p)
            next_s2 = _sample_top_p(s2_logits, top_p)

            gen_s1_list.append(next_s1)
            gen_s2_list.append(next_s2)

            ctx_s1 = torch.cat([ctx_s1[:, 1:], next_s1], dim=1)
            ctx_s2 = torch.cat([ctx_s2[:, 1:], next_s2], dim=1)

        gen_s1 = torch.cat(gen_s1_list, dim=1)
        gen_s2 = torch.cat(gen_s2_list, dim=1)
        return gen_s1, gen_s2


def _sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices.gather(-1, sampled)
    return next_token
