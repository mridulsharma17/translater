import os
import sys
import torch
from huggingface_hub import snapshot_download
import whisper

def download_models():
    """
    Downloads and caches Speech-to-Text Whisper models locally.
    """
    print("=" * 60)
    print("  Let's Talk Voice Agent - Model Pre-loader & Initializer")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Download English Whisper Tiny
    asr_en_repo = os.getenv("ASR_MODEL_EN", "openai/whisper-tiny")
    local_path_en = os.path.join(models_dir, "whisper-tiny")
    print(f"\n[1/2] Checking English Whisper Model ({asr_en_repo})...")
    if not os.path.exists(local_path_en) or not os.listdir(local_path_en):
        print(f"Downloading {asr_en_repo} to {local_path_en}...")
        snapshot_download(repo_id=asr_en_repo, local_dir=local_path_en)
    else:
        print(f"[SUCCESS] English Whisper already exists at: {local_path_en}")

    # Load tiny model into cache
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Testing load for English Whisper on {device}...")
    _ = whisper.load_model("tiny", download_root=local_path_en, device=device)
    print("[SUCCESS] English Whisper Model Loaded Successfully!")

    # 2. Download Hindi Whisper Tiny
    asr_hi_repo = os.getenv("ASR_MODEL_HI", "collabora/whisper-tiny-hindi")
    local_path_hi = os.path.join(models_dir, "whisper-tiny-hi")
    print(f"\n[2/2] Checking Hindi Whisper Model ({asr_hi_repo})...")
    if not os.path.exists(local_path_hi) or not os.listdir(local_path_hi):
        print(f"Downloading {asr_hi_repo} to {local_path_hi}...")
        snapshot_download(repo_id=asr_hi_repo, local_dir=local_path_hi)
    else:
        print(f"[SUCCESS] Hindi Whisper already exists at: {local_path_hi}")

    print("\n" + "=" * 60)
    print("[DONE] All STT models downloaded and ready!")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
