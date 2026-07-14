"""Project-wide environment configuration."""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env(*, override: bool = False) -> bool:
    """Load ``<repository>/.env`` while preserving exported variables by default."""
    return load_dotenv(PROJECT_ROOT / ".env", override=override)

