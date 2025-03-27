from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_TASK_1_DIR = DATA_DIR / "pan25-generative-ai-detection-task1-train"
DATA_TASK_2_DIR = DATA_DIR / "pan25-generative-ai-detection-task2-train"
