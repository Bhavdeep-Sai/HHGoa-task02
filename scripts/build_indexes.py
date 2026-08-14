import sys
import os
import json
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.embeddings import get_embedding_provider
from backend.app.vector_store import get_qdrant_store, PASSAGES_COLLECTION, QA_COLLECTION
from backend.app.vector_store.schema import PointRecord
from backend.app.chunking import ParentChildChunker, SemanticChunker, PassageChunker


SAMPLE_INDIC_DATA = [
    {
        "query_id": 101,
        "query": "Manhattan project successful hone ke baad kya hua?",
        "answer": "The Manhattan Project produced the first nuclear weapons, leading to the atomic bombings of Hiroshima and Nagasaki in August 1945.",
        "language": "hi",
        "target_lang": "hin_Deva",
        "passages": [
            "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. The successful test at Trinity led directly to the atomic bombings of Hiroshima and Nagasaki in August 1945.",
            "Following the conclusion of World War II, the Manhattan Project transitioned into the United States Atomic Energy Commission in 1947 to oversee nuclear research during the Cold War."
        ],
        "is_selected": [1, 0]
    },
    {
        "query_id": 102,
        "query": "ఈ project యొక్క immediate impact ఏమిటి?",
        "answer": "The immediate impact of the project was the acceleration of Allied victory in World War II and the beginning of the atomic age.",
        "language": "te",
        "target_lang": "tel_Telu",
        "passages": [
            "The immediate impact of the project was demonstrating nuclear weapons technology, which accelerated the end of World War II in the Pacific Theater. It established the United States as a global nuclear superpower.",
            "In addition to military applications, the project spurred technological advances in computing, radiation protection, and nuclear medicine."
        ],
        "is_selected": [1, 0]
    },
    {
        "query_id": 103,
        "query": "வாஷிங்டன் நகரம் எப்போது நிறுவப்பட்டது?",
        "answer": "Washington, D.C. was founded on July 16, 1790 after the Residence Act approved the creation of a capital district.",
        "language": "ta",
        "target_lang": "tam_Taml",
        "passages": [
            "Washington, D.C., formally the District of Columbia, was founded on July 16, 1790. The Residence Act approved the creation of a capital district located along the Potomac River on the country's East Coast.",
            "The city was named in honor of George Washington, the first President of the United States and Founding Father."
        ],
        "is_selected": [1, 0]
    },
    {
        "query_id": 104,
        "query": "भारत का संविधान कब लागू हुआ था?",
        "answer": "भारत का संविधान 26 जनवरी 1950 को लागू हुआ था।",
        "language": "hi",
        "target_lang": "hin_Deva",
        "passages": [
            "भारत का संविधान 26 नवंबर 1949 को पारित हुआ और 26 जनवरी 1950 को प्रभावी हुआ। इस दिन को भारत में गणतंत्र दिवस के रूप में मनाया जाता है।",
            "संविधान सभा के अध्यक्ष डॉ. राजेंद्र प्रसाद थे और प्रारूप समिति के अध्यक्ष डॉ. बी.आर. अंबेडकर थे।"
        ],
        "is_selected": [1, 0]
    },
    {
        "query_id": 105,
        "query": "What were the primary consequences of the Manhattan Project?",
        "answer": "Primary consequences included ending World War II, launching the global nuclear arms race, and founding modern nuclear medicine.",
        "language": "en",
        "target_lang": "eng_Latn",
        "passages": [
            "Primary consequences included the immediate surrender of Japan in 1945, initiating the geopolitical nuclear arms race during the Cold War, and founding peacetime nuclear energy research.",
            "Environmental and health impacts were also observed near testing sites such as Trinity Site in New Mexico."
        ],
        "is_selected": [1, 0]
    }
]


def build_sample_indexes():
    print(f"Building Qdrant & BM25 indexes (SAMPLE_MODE = {settings.SAMPLE_MODE})...")
    embeddings = get_embedding_provider()
    store = get_qdrant_store()
    store.init_collections(vector_size=embeddings.dimension)

    passage_records: List[PointRecord] = []
    qa_records: List[PointRecord] = []

    parent_child_chunker = ParentChildChunker()
    semantic_chunker = SemanticChunker()

    for item in SAMPLE_INDIC_DATA:
        query_id = item["query_id"]
        lang = item["language"]
        query_txt = item["query"]
        ans_txt = item["answer"]

        # Index A: Query/Answer Searchable Unit
        qa_representation = f"[QUERY] {query_txt} [ANSWER] {ans_txt}"
        qa_vec = embeddings.embed_text(qa_representation)
        qa_records.append(
            PointRecord(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"qa_{query_id}")),
                vector=qa_vec,
                payload={
                    "query_id": query_id,
                    "language": lang,
                    "document_type": "qa_unit",
                    "chunk_type": "qa",
                    "text": qa_representation,
                    "query": query_txt,
                    "answer": ans_txt,
                    "is_selected": True
                }
            )
        )

        # Index B & C: Passages multi-resolution chunking
        for pass_idx, passage_text in enumerate(item["passages"]):
            is_sel = bool(item["is_selected"][pass_idx])
            meta = {
                "query_id": query_id,
                "passage_id": pass_idx,
                "language": lang,
                "source_language": item["target_lang"],
                "is_selected": is_sel,
                "answer": ans_txt if is_sel else None
            }

            chunks = parent_child_chunker.chunk(passage_text, meta)
            chunks.extend(semantic_chunker.chunk(passage_text, meta))

            for chunk in chunks:
                vec = embeddings.embed_text(chunk.text)
                passage_records.append(
                    PointRecord(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                        vector=vec,
                        payload={
                            "query_id": query_id,
                            "passage_id": pass_idx,
                            "chunk_id": chunk.chunk_id,
                            "parent_id": chunk.parent_id,
                            "chunk_type": chunk.chunk_type,
                            "language": lang,
                            "is_selected": is_sel,
                            "document_type": "passage",
                            "text": chunk.text,
                            "parent_text": passage_text,
                            "answer": chunk.metadata.get("answer")
                        }
                    )
                )

    # Upsert to Qdrant
    store.upsert_points(PASSAGES_COLLECTION, passage_records)
    store.upsert_points(QA_COLLECTION, qa_records)

    # Re-index BM25 in-memory retriever
    from backend.app.retrieval.bm25 import get_bm25_retriever
    bm25_docs = []
    for r in passage_records + qa_records:
        bm25_docs.append({
            "id": r.id,
            "text": r.payload.get("text", ""),
            "payload": r.payload
        })
    get_bm25_retriever().index_documents(bm25_docs)

    print(f"Indexed {len(passage_records)} passage chunks into '{PASSAGES_COLLECTION}'.")
    print(f"Indexed {len(qa_records)} QA units into '{QA_COLLECTION}'.")


if __name__ == "__main__":
    build_sample_indexes()

