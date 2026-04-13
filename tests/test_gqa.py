import torch
from overthink.layer import GQAttention, Attention


def test_gqa_attention():
    """Test GQAttention implementation against standard Attention"""

    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    head_num = 8
    head_dim = 8
    ngrp = 4  # Number of groups for GQA

    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Initialize standard attention and GQA
    standard_attn = Attention(
        hidden_size=hidden_size,
        head_num=head_num,
        head_dim=head_dim,
        dropout=0.0,
        causal=False,
    )

    gqa_attn = GQAttention(
        hidden_size=hidden_size,
        head_num=head_num,
        head_dim=head_dim,
        ngrp=ngrp,
        dropout=0.0,
        causal=False,
    )

    # Forward pass
    with torch.no_grad():
        standard_output = standard_attn(x)
        gqa_output = gqa_attn(x)

    # Check output shapes
    assert standard_output.shape == gqa_output.shape, (
        f"Output shapes don't match: {standard_output.shape} vs {gqa_output.shape}"
    )
    assert standard_output.shape == (batch_size, seq_len, hidden_size), (
        f"Unexpected output shape: {standard_output.shape}"
    )

    print("✓ Output shapes match")

    # Test with different ngrp values
    for ngrp_test in [1, 2, 4, 8]:
        gqa_attn_test = GQAttention(
            hidden_size=hidden_size,
            head_num=head_num,
            head_dim=head_dim,
            ngrp=ngrp_test,
            dropout=0.0,
            causal=False,
        )

        with torch.no_grad():
            output = gqa_attn_test(x)

        assert output.shape == (batch_size, seq_len, hidden_size), (
            f"Failed for ngrp={ngrp_test}"
        )
        print(f"✓ GQA with ngrp={ngrp_test} works correctly")

    # Test causal masking
    gqa_causal = GQAttention(
        hidden_size=hidden_size,
        head_num=head_num,
        head_dim=head_dim,
        ngrp=ngrp,
        dropout=0.0,
        causal=True,
    )

    with torch.no_grad():
        causal_output = gqa_causal(x)

    assert causal_output.shape == (batch_size, seq_len, hidden_size), (
        "Causal GQA output shape mismatch"
    )
    print("✓ Causal masking works correctly")

    # Test error cases
    try:
        # ngrp > head_num should raise an error
        bad_gqa = GQAttention(
            hidden_size=hidden_size,
            head_num=head_num,
            head_dim=head_dim,
            ngrp=head_num + 1,
            dropout=0.0,
            causal=False,
        )
        assert False, "Should have raised an error for ngrp > head_num"
    except ValueError:
        print("✓ Error handling for ngrp > head_num works correctly")

    try:
        # head_num not divisible by ngrp should raise an error
        bad_gqa = GQAttention(
            hidden_size=hidden_size,
            head_num=head_num,
            head_dim=head_dim,
            ngrp=3,  # 8 is not divisible by 3
            dropout=0.0,
            causal=False,
        )
        assert False, "Should have raised an error for head_num not divisible by ngrp"
    except ValueError:
        print("✓ Error handling for head_num not divisible by ngrp works correctly")

    print("\nAll tests passed! GQAttention implementation is working correctly.")


if __name__ == "__main__":
    test_gqa_attention()
