import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Importing app.main must not require a real database.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
