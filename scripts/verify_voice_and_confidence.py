import asyncio
import os
import sys
import json

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.orchestrator import RAGOrchestrator
from backend.app.embeddings import get_embedding_provider
from backend.app.stt.mock_stt import MockSTTProvider


async def run_verification():
    print("=" * 80)
    print("FINAL VERIFICATION: VOICE & CONFIDENCE CALIBRATION AUDIT")
    print("=" * 80)

    embeddings = get_embedding_provider()
    embeddings.warmup()
    
    orchestrator = RAGOrchestrator()

    test_cases = [
        {
            "code": "A",
            "name": "Exact Manhattan Project Query",
            "type": "text",
            "query": "What was the immediate impact of the success of the Manhattan Project?"
        },
        {
            "code": "B",
            "name": "Casual Greeting",
            "type": "text",
            "query": "Hello, how are you?"
        },
        {
            "code": "C",
            "name": "Capital of India Query",
            "type": "text",
            "query": "What is the capital of India?"
        },
        {
            "code": "D",
            "name": "Normal Indian-Language Question (Hindi)",
            "type": "text",
            "query": "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?"
        },
        {
            "code": "E (Voice Corruption)",
            "name": "Corrupted Voice Acoustic Utterance ('one-hat project')",
            "type": "voice",
            "query": "The immediate effect of the success of the one-hat project."
        }
    ]

    for tc in test_cases:
        print(f"\n[TEST {tc['code']}] {tc['name']}")
        print(f"Query: \"{tc['query']}\" (Type: {tc['type']})")
        
        if tc["type"] == "voice":
            audio_bytes = b"RIFF____WAVEfmt " + b"\x00" * 3000
            orchestrator.stt_provider = MockSTTProvider(preset_text=tc["query"])
            res = await orchestrator.execute_voice_query(audio_bytes=audio_bytes, filename="voice_input.wav")
        else:
            res = await orchestrator.execute_text_query(query=tc["query"])

        safe_ans = res.answer.encode('ascii', 'replace').decode('ascii')
        print(f"  Exact Transcription: \"{res.query}\"")
        print(f"  Intent:              {res.intent}")
        print(f"  Grounded:            {res.grounded}")
        print(f"  Final Confidence:    {res.confidence:.2%}")
        print(f"  Answer:              \"{safe_ans}\"")
        print(f"  Latencies:           STT: {res.stage_latencies.stt_latency_ms} ms | Ret: {res.stage_latencies.retrieval_latency_ms} ms | RAG: {res.stage_latencies.rag_latency_ms} ms | Total: {res.stage_latencies.total_latency_ms} ms")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_verification())
