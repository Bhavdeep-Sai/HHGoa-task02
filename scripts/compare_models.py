import asyncio
import os
import sys
import io
import wave
import time
import numpy as np

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.stt.sarvam import SarvamSTTProvider
from scripts.benchmark_stt_profiling import generate_sample_wav


async def compare_all_sarvam_models():
    print("=" * 80)
    print("COMPARISON OF ALL SUPPORTED SARVAM ASR MODELS")
    print("=" * 80)

    test_audio = generate_sample_wav(duration_sec=1.5)
    supported_models = [
        "saaras:v3",
        "saarika:v2.5",
        "saarika:v2",
        "saarika:flash",
        "saaras:v4"
    ]

    for model_name in supported_models:
        provider = SarvamSTTProvider(model=model_name)
        latencies = []
        errs = 0

        for r in range(2):
            try:
                txt, lang, lat = await provider.transcribe(test_audio, filename="test.wav", language_hint="en-IN")
                latencies.append(lat)
            except Exception as e:
                errs += 1

        if latencies:
            avg_l = np.mean(latencies)
            min_l = np.min(latencies)
            max_l = np.max(latencies)
            print(f"Model '{model_name:<15}': Avg = {avg_l:>7.2f} ms | Min = {min_l:>7.2f} ms | Max = {max_l:>7.2f} ms | Errors = {errs}")
        else:
            print(f"Model '{model_name:<15}': FAILED ({errs} errors)")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(compare_all_sarvam_models())
