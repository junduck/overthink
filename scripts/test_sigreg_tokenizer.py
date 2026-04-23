"""End-to-end test for SIGRegTokenizer training pipeline.

Uses real market data from /Volumes/CN_book/cn_1m_ohlcv.
Runs on MPS (Apple Silicon) or CUDA with a tiny model and 3 codes.

Usage:
    uv run python scripts/test_sigreg_tokenizer.py
"""

import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from overthink.data.dataset_window import SlidingWindowDataset
from overthink.model.sigreg_tokenizer import SIGRegTokenizer
from overthink.model.sigreg_tokenizer_config import SIGRegTokenizerConfig
from torch.utils.data import DataLoader

SQLITE_DIR = "/Volumes/CN_book/cn_1m_ohlcv_sqlite"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CFG = SIGRegTokenizerConfig(
    d_model=64, ff_dim=128, n_enc_layers=1, n_dec_layers=1,
    s1_bits=10, s2_bits=10,
    sigreg_num_slices=8, sigreg_lambda=0.05,
)
MAX_CODES = 10
N_TRAIN_STEPS = 10


def collate_ohlcv(batch):
    return torch.stack(batch)


def test_data_loading():
    print("  test: dataset loads from real data")
    ds = SlidingWindowDataset(SQLITE_DIR, window_size=64, stride=30, max_codes=MAX_CODES)
    samples = []
    for ohlcv in ds:
        samples.append(ohlcv)
        if len(samples) >= 20:
            break
    assert len(samples) > 0, "no samples yielded"
    for s in samples:
        assert s.dim() == 2 and s.size(1) == 6 and s.size(0) == 64
    print(f"    {len(samples)} samples, all shape={samples[0].shape}")
    print("    PASS")


def test_forward_pass():
    print("  test: forward pass shapes on device")
    model = SIGRegTokenizer(CFG).to(DEVICE)
    x = torch.randn(4, 32, 6, device=DEVICE)
    (recon_s1, recon_full), sigreg_loss, quantized, indices = model(x)
    assert recon_s1.shape == (4, 32, 6)
    assert recon_full.shape == (4, 32, 6)
    assert sigreg_loss.dim() == 0
    assert quantized.shape == (4, 32, 20)
    assert indices.shape == (4, 32)
    assert quantized.min() >= -1 and quantized.max() <= 1
    print("    PASS")


def test_encode_decode():
    print("  test: encode/decode roundtrip on device")
    model = SIGRegTokenizer(CFG).to(DEVICE)
    model.eval()
    x = torch.randn(2, 16, 6, device=DEVICE)
    s1, s2 = model.encode(x, half=True)
    assert s1.shape == (2, 16) and s1.min() >= 0 and s1.max() < 1024
    assert s2.shape == (2, 16) and s2.min() >= 0 and s2.max() < 1024
    decoded = model.decode((s1, s2), half=True)
    assert decoded.shape == (2, 16, 6)

    s1_b, s2_b = model.encode(x, half=True)
    assert torch.equal(s1, s1_b), "non-deterministic encode"
    assert torch.equal(s2, s2_b), "non-deterministic encode"
    print("    PASS (roundtrip + determinism)")


def test_backward():
    print("  test: backward + gradient flow on device")
    model = SIGRegTokenizer(CFG).to(DEVICE)
    x = torch.randn(4, 32, 6, device=DEVICE)
    (recon_s1, recon_full), sigreg_loss, _, _ = model(x)
    loss = (F.mse_loss(recon_s1, x) + F.mse_loss(recon_full, x) + 0.05 * sigreg_loss) / 2
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no gradients"
    print(f"    loss={loss.item():.4f}")
    print("    PASS")


def test_sigreg_convergence():
    print("  test: SIGReg drives latents toward N(0, I)")
    model = SIGRegTokenizer(CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, 64, 6, device=DEVICE) * 10 + 50

    vals = []
    for step in range(100):
        z = model._encode_latent(x)
        val = model.sigreg(z)
        vals.append(val.item())
        optimizer.zero_grad()
        val.backward()
        optimizer.step()

    before = vals[0]
    after = vals[-1]
    print(f"    {before:.2f} -> {after:.2f} (100 steps, min={min(vals):.2f})")
    assert min(vals) < before, f"SIGReg never decreased: {before:.2f}, min={min(vals):.2f}"
    print("    PASS")


def test_training_with_real_data(ckpt_dir: Path):
    print("  test: training loop with real data + checkpoint")
    model = SIGRegTokenizer(CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ds = SlidingWindowDataset(SQLITE_DIR, window_size=64, stride=30, max_codes=MAX_CODES)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_ohlcv)

    model.train()
    losses = []
    for i, ohlcv in enumerate(loader):
        if i >= N_TRAIN_STEPS:
            break
        ohlcv = ohlcv.to(DEVICE)
        (recon_s1, recon_full), sigreg_loss, _, _ = model(ohlcv)
        recon = F.mse_loss(recon_s1, ohlcv) + F.mse_loss(recon_full, ohlcv)
        loss = (recon + 0.05 * sigreg_loss) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"    step {i+1}: loss={loss.item():.2f} recon={recon.item():.2f} sigreg={sigreg_loss.item():.2f}")

    assert len(losses) == N_TRAIN_STEPS

    torch.save({"config": CFG.model_dump(), "model_state_dict": model.state_dict()}, ckpt_dir / "test.pt")

    loaded = SIGRegTokenizer(SIGRegTokenizerConfig.model_validate(CFG.model_dump())).to(DEVICE)
    loaded.load_state_dict(torch.load(ckpt_dir / "test.pt", weights_only=True)["model_state_dict"])
    loaded.eval()

    x = torch.randn(1, 16, 6, device=DEVICE)
    assert torch.equal(model.encode(x, half=True)[0], loaded.encode(x, half=True)[0])
    print("    checkpoint save/load: PASS")


def test_code_diversity():
    print("  test: tokens diversify after training")
    model = SIGRegTokenizer(CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(16, 32, 6, device=DEVICE) * 10 + 50

    model.train()
    for _ in range(30):
        (r1, r2), sr, _, _ = model(x)
        loss = (F.mse_loss(r1, x) + F.mse_loss(r2, x) + 0.05 * sr) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    s1, s2 = model.encode(x, half=True)
    u1, u2 = len(s1.unique()), len(s2.unique())
    print(f"    unique s1={u1} s2={u2} out of {s1.numel()} positions")
    assert u1 > 1, "s1 all identical"
    assert u2 > 1, "s2 all identical"
    print("    PASS")


def main():
    print("=" * 60)
    print("SIGRegTokenizer Pipeline Test")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    if not Path(SQLITE_DIR).exists():
        print(f"\nSKIP: {SQLITE_DIR} not found")
        return

    ckpt_dir = Path(tempfile.mkdtemp(prefix="sigreg_test_"))
    try:
        print("\n1. Data loading")
        test_data_loading()

        print("\n2. Forward/backward")
        test_forward_pass()
        test_backward()

        print("\n3. Encode/decode + determinism")
        test_encode_decode()

        print("\n4. SIGReg convergence")
        test_sigreg_convergence()

        print("\n5. Training on real data")
        test_training_with_real_data(ckpt_dir)

        print("\n6. Code diversity")
        test_code_diversity()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
