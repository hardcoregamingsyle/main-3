"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    tmp = Path(tempfile.mkdtemp(prefix="moe_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.num_hidden_layers = 4
    model.config.num_experts = 8
    model.config.num_experts_per_tok = 2
    model.config.hidden_size = 512
    model.config.intermediate_size = 1024
    model.config.vocab_size = 32000
    model.dtype = torch.float16
    model.device = torch.device("cpu")
    return model


@pytest.fixture
def sample_input_ids():
    """Sample input tensor."""
    return torch.randint(0, 32000, (1, 128), dtype=torch.long)


@pytest.fixture
def sample_kv_cache():
    """Sample KV cache."""
    return [
        (torch.randn(1, 8, 128, 64), torch.randn(1, 8, 128, 64))
        for _ in range(4)
    ]


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
