import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import uuid
import time
import logging
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.embeddings import get_embedding_provider
from backend.app.vector_store import get_vector_store, PASSAGES_COLLECTION, QA_COLLECTION
from backend.app.chunking import ParentChildChunker, SemanticChunker
from backend.app.retrieval.bm25 import get_bm25_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MSMARCO_Ingestion")

CHECKPOINT_FILE = os.path.join("data", "checkpoints", "msmarco_checkpoint.json")
MANIFEST_FILE = os.path.join("data", "index_manifest.json")
SENTINEL_QID = 1185869


def safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, np.ndarray)):
        if len(val) > 0:
            return safe_str(val[0])
        return ""
    return str(val).strip()


def extract_record_passages_and_qa(
    row: Dict[str, Any],
    split_name: str,
    lang: str,
    parent_chunker: ParentChildChunker,
    semantic_chunker: SemanticChunker
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qid = int(row.get("query_id") or 0)
    query_hi = safe_str(row.get("query"))
    query_en = safe_str(row.get("Eng_Query")) or query_hi
    answer_hi = safe_str(row.get("Answer"))
    answer_en = safe_str(row.get("Eng_Answer")) or answer_hi

    # Clean leading formatting if present (e.g. ")what was...")
    if query_en.startswith(")") or query_en.startswith("("):
        query_en = query_en[1:].strip()

    passages_obj = row.get("passages", {})
    raw_passages = []
    is_selected_flags = []

    if isinstance(passages_obj, dict):
        raw_passages = passages_obj.get("passage_text", []) or []
        is_selected_flags = passages_obj.get("is_selected", []) or []

    # If raw passages list is empty (common in train QA records), construct canonical passage from Answer
    if not raw_passages and answer_en:
        raw_passages = [answer_en]
        is_selected_flags = [1]
        if answer_hi and answer_hi != answer_en:
            raw_passages.append(answer_hi)
            is_selected_flags.append(1)

    passage_chunks = []
    for p_idx, p_text in enumerate(raw_passages):
        p_text_str = safe_str(p_text)
        if not p_text_str or len(p_text_str) < 10:
            continue

        is_sel = bool(is_selected_flags[p_idx]) if p_idx < len(is_selected_flags) else False
        parent_id = f"{qid}_p{p_idx}"

        doc_meta = {
            "query_id": qid,
            "parent_id": parent_id,
            "passage_id": p_idx,
            "language": lang,
            "is_selected": is_sel,
            "dataset": "ai4bharat/MSMARCO-XI",
            "split": split_name,
            "query_en": query_en,
            "query_hi": query_hi,
            "parent_text": p_text_str
        }

        # 1. Semantic Chunking
        sem_chunks = semantic_chunker.chunk(p_text_str, doc_meta)
        for s_idx, chk in enumerate(sem_chunks):
            passage_chunks.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"msmarco_{split_name}_{qid}_{parent_id}_sem_{s_idx}")),
                "text": chk.text,
                "payload": {
                    **doc_meta,
                    "chunk_id": chk.chunk_id,
                    "chunk_type": "semantic",
                    "text": chk.text
                }
            })

        # 2. Parent-Child Chunking
        pc_chunks = parent_chunker.chunk(p_text_str, doc_meta)
        for pc_idx, chk in enumerate(pc_chunks):
            passage_chunks.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"msmarco_{split_name}_{qid}_{chk.chunk_id}")),
                "text": chk.text,
                "payload": {
                    **doc_meta,
                    "chunk_id": chk.chunk_id,
                    "chunk_type": chk.chunk_type,
                    "text": chk.text
                }
            })

    # QA Unit Record
    qa_unit = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"msmarco_{split_name}_qa_{qid}")),
        "text": f"Question: {query_en}\nAnswer: {answer_en}",
        "payload": {
            "query_id": qid,
            "query": query_en,
            "query_hi": query_hi,
            "answer": answer_en,
            "answer_hi": answer_hi,
            "language": lang,
            "dataset": "ai4bharat/MSMARCO-XI",
            "split": split_name,
            "chunk_type": "qa"
        }
    }

    return passage_chunks, qa_unit


def run_stratified_ingestion(sample_size_per_split: int = 400):
    logger.info("=" * 65)
    logger.info("STARTING STRATIFIED MSMARCO-XI PRODUCTION INGESTION")
    logger.info("=" * 65)

    embeddings = get_embedding_provider()
    embeddings.warmup()
    vector_store = get_vector_store()
    bm25 = get_bm25_retriever()

    parent_chunker = ParentChildChunker()
    semantic_chunker = SemanticChunker()

    # 1. Fetch & Ingest Sentinel QID 1185869 from train/hintrain.parquet
    logger.info(f"Targeting Sentinel QID {SENTINEL_QID} from 'train/hintrain.parquet'...")
    train_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="train/hintrain.parquet", repo_type="dataset")
    pf_train = pq.ParquetFile(train_path)

    sentinel_found = False
    sentinel_record_data = None
    all_passage_chunks = []
    all_qa_units = []
    processed_qids = set()

    for batch in pf_train.iter_batches(batch_size=20000):
        qids = batch.column("query_id").to_pylist()
        if SENTINEL_QID in qids:
            idx = qids.index(SENTINEL_QID)
            row = {col: batch.column(col)[idx].as_py() for col in batch.schema.names}
            chunks, qa = extract_record_passages_and_qa(row, "train", "hi", parent_chunker, semantic_chunker)
            all_passage_chunks.extend(chunks)
            all_qa_units.append(qa)
            processed_qids.add(SENTINEL_QID)
            sentinel_found = True
            sentinel_record_data = {
                "query_id": SENTINEL_QID,
                "query_en": safe_str(row.get("Eng_Query")),
                "answer_en": safe_str(row.get("Eng_Answer")),
                "passages_count": len(chunks)
            }
            logger.info(f"✓ Sentinel QID {SENTINEL_QID} extracted: '{sentinel_record_data['query_en']}' -> {len(chunks)} chunks")
            break

    # 2. Ingest Stratified Validation Records (Hindi, Telugu)
    splits_to_ingest = [
        ("validation/hinval.parquet", "validation", "hi", 500),
        ("validation/telval.parquet", "validation", "te", 300),
        ("train/hintrain.parquet", "train", "hi", 200)
    ]

    for file_name, split_type, lang, target_count in splits_to_ingest:
        logger.info(f"Ingesting {target_count} records from '{file_name}' (lang={lang})...")
        try:
            split_file_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename=file_name, repo_type="dataset")
            pf = pq.ParquetFile(split_file_path)
            count_collected = 0
            for batch in pf.iter_batches(batch_size=5000):
                names = batch.schema.names
                num_rows = batch.num_rows
                for r_idx in range(num_rows):
                    row = {col: batch.column(col)[r_idx].as_py() for col in names}
                    qid = int(row.get("query_id") or 0)
                    if qid in processed_qids:
                        continue

                    chunks, qa = extract_record_passages_and_qa(row, split_type, lang, parent_chunker, semantic_chunker)
                    all_passage_chunks.extend(chunks)
                    all_qa_units.append(qa)
                    processed_qids.add(qid)
                    count_collected += 1

                    if count_collected >= target_count:
                        break
                if count_collected >= target_count:
                    break
            logger.info(f"  Collected {count_collected} records from {file_name}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

    total_records = len(processed_qids)
    total_passage_chunks = len(all_passage_chunks)
    logger.info(f"Total Records Collected: {total_records} | Total Passage Chunks: {total_passage_chunks}")

    # 3. Batch Embed and Index into FastVectorStore
    logger.info("Generating embeddings and building FastVectorStore matrices...")
    t0_embed = time.perf_counter()

    # Passages
    p_texts = [c["text"] for c in all_passage_chunks]
    p_ids = [c["id"] for c in all_passage_chunks]
    p_payloads = [c["payload"] for c in all_passage_chunks]
    
    batch_size = 64
    p_vectors = []
    for b_start in range(0, len(p_texts), batch_size):
        b_end = min(b_start + batch_size, len(p_texts))
        b_vecs = embeddings.embed_batch(p_texts[b_start:b_end])
        p_vectors.extend(b_vecs)
        if b_start % 2000 == 0 and b_start > 0:
            logger.info(f"  Embedded {b_start}/{len(p_texts)} passages...")

    vector_store.upsert_batch(PASSAGES_COLLECTION, p_ids, p_vectors, p_payloads)

    # QA Units
    qa_texts = [q["text"] for q in all_qa_units]
    qa_ids = [q["id"] for q in all_qa_units]
    qa_payloads = [q["payload"] for q in all_qa_units]
    qa_vectors = embeddings.embed_batch(qa_texts)
    vector_store.upsert_batch(QA_COLLECTION, qa_ids, qa_vectors, qa_payloads)

    # Persist vector store matrices to disk
    vector_store.save_to_disk()
    embed_total_s = time.perf_counter() - t0_embed
    logger.info(f"✓ FastVectorStore built and saved in {embed_total_s:.2f}s ({len(p_vectors)} passage vectors, {len(qa_vectors)} QA vectors)")

    # 4. Build and Persist High-Performance BM25 Inverted Posting Lists
    logger.info("Building BM25 sparse posting lists...")
    bm25_documents = []
    for c in all_passage_chunks:
        bm25_documents.append({
            "id": c["id"],
            "text": c["text"],
            "payload": c["payload"]
        })
    for q in all_qa_units:
        bm25_documents.append({
            "id": q["id"],
            "text": q["text"],
            "payload": q["payload"]
        })

    bm25.index_documents(bm25_documents)
    bm25_storage_path = getattr(settings, "BM25_STORAGE_PATH", "./data/bm25_index.pkl")
    bm25.save_to_disk(bm25_storage_path)
    logger.info(f"✓ BM25 posting lists indexed ({len(bm25_documents)} documents) and saved to {bm25_storage_path}")

    # 5. Write Checkpoint and Manifest
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    checkpoint_data = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "split": "stratified_train_val",
        "records_processed": total_records,
        "vectors_indexed": len(p_vectors) + len(qa_vectors),
        "sentinel_found": sentinel_found,
        "sentinel_query_id": SENTINEL_QID,
        "sentinel_record": sentinel_record_data,
        "processed_query_ids": list(processed_qids),
        "completed": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    manifest_data = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "version": "1.0.0",
        "build_timestamp": datetime.utcnow().isoformat(),
        "splits": ["train", "validation"],
        "languages": ["en", "hi", "te"],
        "record_count": total_records,
        "passage_vector_count": len(p_vectors),
        "qa_vector_count": len(qa_vectors),
        "total_vector_count": len(p_vectors) + len(qa_vectors),
        "bm25_corpus_count": len(bm25_documents),
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_dim": embeddings.dimension,
        "sentinel_query_id": SENTINEL_QID,
        "sentinel_verified": sentinel_found,
        "chunking_strategies": ["Parent-Child (150/600)", "Semantic (300)"],
        "vector_engine": "FastVectorStore (BLAS Matrix Dot-Product)",
        "lexical_engine": "BM25 Inverted Posting Lists"
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"✓ Manifest saved to {MANIFEST_FILE}")

    logger.info("=" * 65)
    logger.info("INGESTION & INDEXING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_stratified_ingestion()
