# Pytest configuration and fixtures for MoE Ultra Engine
"""Shared test fixtures and configurations."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Set up test environment variables before imports
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("API_HOST", "0.0.0.0")
os.environ.setdefault("API_PORT", "8000")


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = MagicMock()
    config.api.host = "0.0.0.0"
    config.api.port = 8000
    config.api.debug = False
    config.model.name = "qwen-2.4t-moe"
    config.model.context_window = 32768
    config.model.max_tokens = 8192
    config.model.device = "cuda"
    config.database.url = "sqlite:///test.db"
    config.logging.level = "DEBUG"
    config.security.jwt_secret = "test-secret-key-change-in-production"
    config.security.rate_limit = 100
    return config


@pytest.fixture
def mock_llm_engine():
    """Mock LLM engine for testing API endpoints."""
    engine = MagicMock()
    engine.generate.return_value = {
        "tokens": ["Hello", "world"],
        "logprobs": [0.9, 0.85],
        "completion_time": 0.15,
        "prompt_tokens": 10,
        "completion_tokens": 2,
    }
    engine.is_loaded.return_value = True
    engine.load_model.return_value = True
    return engine


@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return {
        "role": "user",
        "content": "Hello, how can you help me?",
        "timestamp": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_response():
    """Sample model response for testing."""
    return {
        "id": "msg_abc123",
        "model": "qwen-2.4t-moe",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I can help you with various tasks including code generation, analysis, and more.",
                },
                "finish_reason": "stop",
            }
        ],
        "created": 1705312200,
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }


@pytest.fixture
def valid_api_key():
    """Valid test API key."""
    return "sk-test-1234567890abcdef"


@pytest.fixture
def api_headers(valid_api_key):
    """Standard API headers for authenticated requests."""
    return {
        "Authorization": f"Bearer {valid_api_key}",
        "Content-Type": "application/json",
        "X-Request-ID": "test-request-123",
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for each test."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr("core.config.Config.instance", MagicMock())


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    db_path = tmp_path / "test.db"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def client_with_auth(client, api_headers):
    """Test client with authentication headers."""
    for header, value in api_headers.items():
        client.headers[header] = value
    return client


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (for CI only)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Skip slow/integration/e2e tests unless --run-slow flag is provided."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow flag")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Add timing information to test results."""
    import time

    start = time.time()
    yield
    duration = time.time() - start
    item.add_marker(pytest.mark.duration(seconds=duration))
