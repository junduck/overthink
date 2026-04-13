# OVERTHINK

## Build, Run, and Test Commands

### Package Management (uv)
```sh
# Install dependencies
uv sync

# Run examples
uv run python examples/example.py
uv run python examples/example_film.py

# Run the notebook example
uv run jupyter notebook examples/example.ipynb
```

### Testing
```sh
# Run all tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_gqa.py
```

### Linting and Type Checking
```sh
# Run pylint on the package
uv run pylint src/overthink/
```

### Development
```sh
# Create virtual environment
uv venv

# Install package in editable mode
uv pip install -e .
```

---

## Code Style Guidelines

### Python Version and Type Hints
- Target Python 3.13+
- Always use **type hints** for function signatures
- Use `from typing import Optional, Tuple, Literal` for complex types
- Avoid `Any` unless absolutely necessary

### Imports
Organize imports in three blocks, separated by blank lines:
```python
# Standard library
import math
from typing import Optional, Tuple

# Third-party packages
import torch
from einops import repeat
from torch import nn

# Local application imports
from overthink.block import TransBlock, TransStack
from overthink.layer import Attention, RoPE
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `OverthinkModel`, `TransBlock`)
- **Functions/variables**: `snake_case` (e.g., `input_projection`, `high_freq_step`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_EPS`)
- **Private methods**: prefix with `_` (e.g., `_do_feature_map`)

### Docstrings
Use Google-style docstrings for all public classes and functions:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Short description.

    Longer description if needed.

    Args:
        x: Input tensor [B, S, D]
        dropout: Dropout probability (default: 0.0)

    Returns:
        Output tensor [B, S, D]
    """
```
- Document tensor shapes using brackets: `[B, S, D]` (Batch, Seq, Dim)
- Keep docstrings concise; avoid trivial descriptions

### Error Handling
- Use `ValueError` for invalid argument values
- Use `NotImplementedError` for unimplemented features
- Use `AssertionError` sparingly for internal invariants:
```python
# For invalid arguments (public API)
if ngrp > head_num:
    raise ValueError(f"ngrp ({ngrp}) cannot exceed head_num ({head_num})")

# For internal invariants
assert self.temporal_mix is not None
```

### Configuration (Pydantic)
- Use `pydantic.BaseModel` for all configuration classes
- Use `Field()` with `description` for all fields
- Use `model_validator` for cross-field validation
- Use `Literal` types for enum-like choices

### PyTorch Best Practices
- Always call `super().__init__()` in nn.Module subclasses
- Use `torch.no_grad()` context for inference
- Use `torch.no_grad()` for most reasoning loop iterations in inference
- Use `einops.rearrange` for tensor reshaping; document pattern strings
- Prefer `F.scaled_dot_product_attention` over manual attention computation
- Use `nn.Parameter` for learnable parameters

### Tensor Shape Conventions
Document tensor shapes in comments and docstrings:
```
[B, S, D]  - Batch, Sequence, Dimension
[B, H, S, D] - Batch, Heads, Sequence, Head Dim
[B, S, F]  - Batch, Sequence, Features
```

### Testing Guidelines
- Create test functions prefixed with `test_`
- Use simple numbers: 2, 4, 8, 10, 64
- Use `torch.no_grad()` in tests for speed
- Test error cases with `try/except ValueError`
- Print test status with `print("✓ Test passed")`

---

## Coding Principles

1. Prefer straightforward implementation over clever code
2. Avoid type/generic gymnastics unless no alternative
3. Avoid monkey patching
4. Comment business logic, not implementation details
5. Layout logical ownership (who owns business logic, resources, information)
6. Layout data flow (suppliers, consumers, information leakage)
7. Apply YAGNI (You Aren't Gonna Need It) and KISS (Keep It Simple, Stupid)
8. For refactoring: create local dev branch first
9. Pre-1.0: focus on functionality over linter perfection
10. You don't need to always agree with user feedback; provide honest alternatives
