from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
GOLDEN_DIR = ROOT / "data" / "golden"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

DEFAULT_TOP_K = 5
AUTO_INTENT_THRESHOLD = 0.60
AUTO_SIMILARITY_THRESHOLD = 0.45
