import pytest
from backend.app.generation import ConfidenceAwareAnswerRouter


@pytest.mark.asyncio
async def test_fast_path_routing():
    router = ConfidenceAwareAnswerRouter()
    contexts = [
        {
            "id": "qa_101",
            "payload": {
                "document_type": "qa_unit",
                "answer": "The Manhattan Project produced the first nuclear weapons."
            }
        }
    ]

    response, fast_path, gen_ms = await router.route_and_generate(
        query="What did the Manhattan project produce?",
        retrieved_contexts=contexts,
        confidence_score=0.92
    )

    assert fast_path
    assert "nuclear weapons" in response.answer
    assert gen_ms < 10.0
