import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings


def download_or_stream_msmarco():
    """
    Streams or downloads ai4bharat/MSMARCO-XI dataset in batched mode.
    """
    print(f"Dataset streaming module initialized (SAMPLE_MODE={settings.SAMPLE_MODE}).")
    if settings.SAMPLE_MODE:
        print("SAMPLE_MODE=True: Skipping full 55.6GB download. Using streaming batch processor.")
        return None

    try:
        from datasets import load_dataset
        print("Connecting to Hugging Face dataset 'ai4bharat/MSMARCO-XI' in streaming mode...")
        ds = load_dataset("ai4bharat/MSMARCO-XI", streaming=True)
        return ds
    except Exception as e:
        print(f"Could not load streaming dataset directly: {e}")
        return None


if __name__ == "__main__":
    download_or_stream_msmarco()
