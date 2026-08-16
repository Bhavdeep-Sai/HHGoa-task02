import os
import sys
import json
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data"

DB_FILE = DATA_DIR / "msmarco_xi.sqlite"
BACKEND_DB_FILE = BACKEND_DATA_DIR / "msmarco_xi.sqlite"


def safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join([safe_str(v) for v in val if v is not None]).strip()
    return str(val).strip()


def build_sqlite_database():
    print("=" * 70)
    print("BUILDING COMPACT READ-ONLY SQLITE FTS5 RETRIEVAL DATABASE")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DB_FILE.exists():
        try:
            DB_FILE.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(str(DB_FILE))
    cur = conn.cursor()

    # Enable fast insertion mode
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("PRAGMA temp_store = MEMORY")

    # 1. Create Primary documents table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        qid TEXT,
        language TEXT,
        query TEXT,
        answer TEXT,
        passage TEXT,
        source TEXT,
        metadata_json TEXT
    );
    """)

    # 2. Create FTS5 virtual table for full-text search across query, answer, passage
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        query,
        answer,
        passage,
        content='documents',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    );
    """)

    # 3. Create index on QID and language for instant filtering & provenance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_qid ON documents(qid);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_lang ON documents(language);")

    # Load records from existing processed BM25 pickle & metadata
    bm25_source_paths = [
        DATA_DIR / "bm25_index.pkl",
        BACKEND_DATA_DIR / "bm25_index.pkl",
    ]
    bm25_file = None
    for p in bm25_source_paths:
        if p.exists():
            bm25_file = p
            break

    records_to_insert = []
    seen_unique = set()

    if bm25_file and bm25_file.exists():
        print(f"Loading existing processed MSMARCO-XI data from: {bm25_file}")
        with open(bm25_file, "rb") as f:
            bm25_data = pickle.load(f)
        corpus = bm25_data.get("corpus", [])
        print(f"Loaded {len(corpus)} items from corpus.")

        for item in corpus:
            doc_id = item.get("id")
            payload = item.get("payload", {})
            qid = safe_str(payload.get("query_id"))
            lang = safe_str(payload.get("language") or "en")
            
            query_en = safe_str(payload.get("query_en") or payload.get("query"))
            query_hi = safe_str(payload.get("query_hi"))
            
            # Combine English and Indic queries in query field for multilingual searchability
            combined_query_parts = []
            if query_en:
                combined_query_parts.append(query_en)
            if query_hi and query_hi != query_en:
                combined_query_parts.append(query_hi)
            query_field = " | ".join(combined_query_parts) if combined_query_parts else query_en

            answer_en = safe_str(payload.get("answer"))
            answer_hi = safe_str(payload.get("answer_hi"))
            answer_field = answer_en or answer_hi or ""

            parent_text = safe_str(payload.get("parent_text"))
            chunk_text = safe_str(payload.get("text") or item.get("text"))
            passage_field = parent_text or chunk_text or ""

            source = safe_str(payload.get("dataset") or "ai4bharat/MSMARCO-XI")

            # Clean metadata
            meta = {
                "chunk_id": payload.get("chunk_id", doc_id),
                "parent_id": payload.get("parent_id"),
                "passage_id": payload.get("passage_id"),
                "chunk_type": payload.get("chunk_type", "passage"),
                "is_selected": payload.get("is_selected", True),
                "split": payload.get("split", "train"),
                "query_en": query_en,
                "query_hi": query_hi,
                "answer_en": answer_en,
                "answer_hi": answer_hi,
                "parent_text": parent_text,
                "chunk_text": chunk_text
            }

            records_to_insert.append((
                qid,
                lang,
                query_field,
                answer_field,
                passage_field,
                source,
                json.dumps(meta, ensure_ascii=False)
            ))
    else:
        print(f"Warning: {bm25_file} not found!")

    # Insert sample benchmark records if not present
    from scripts.build_indexes import SAMPLE_INDIC_DATA
    sample_qids = {str(item["query_id"]) for item in SAMPLE_INDIC_DATA}
    existing_qids = {r[0] for r in records_to_insert}

    for item in SAMPLE_INDIC_DATA:
        qid = str(item["query_id"])
        lang = item["language"]
        q_txt = item["query"]
        a_txt = item["answer"]
        passages = item.get("passages", [])
        p_txt = passages[0] if passages else ""
        meta = {
            "chunk_id": f"sample_{qid}",
            "parent_id": f"sample_{qid}_p0",
            "passage_id": 0,
            "chunk_type": "passage",
            "is_selected": True,
            "split": "benchmark",
            "query_en": q_txt,
            "query_hi": q_txt,
            "answer_en": a_txt,
            "answer_hi": a_txt,
            "parent_text": p_txt,
            "chunk_text": p_txt
        }
        records_to_insert.append((
            qid,
            lang,
            q_txt,
            a_txt,
            p_txt,
            "ai4bharat/MSMARCO-XI-Sample",
            json.dumps(meta, ensure_ascii=False)
        ))

    print(f"Inserting {len(records_to_insert)} records into documents table...")
    cur.executemany("""
    INSERT INTO documents (qid, language, query, answer, passage, source, metadata_json)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, records_to_insert)

    # Rebuild FTS5 index
    print("Rebuilding SQLite FTS5 index...")
    cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild');")

    # Optimize FTS5 structure
    cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize');")

    conn.commit()

    # Verify counts and sentinel
    cur.execute("SELECT COUNT(*) FROM documents;")
    doc_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM documents_fts;")
    fts_count = cur.fetchone()[0]

    cur.execute("SELECT id, qid, language, query, answer, passage FROM documents WHERE qid = '1185869' LIMIT 1;")
    sentinel_doc = cur.fetchone()

    conn.close()

    db_size_mb = round(DB_FILE.stat().st_size / (1024 * 1024), 2)
    print("=" * 70)
    print("DATABASE BUILD COMPLETE")
    print(f"Database File: {DB_FILE}")
    print(f"Database Size: {db_size_mb} MB")
    print(f"Document Count: {doc_count}")
    print(f"FTS5 Row Count: {fts_count}")
    print(f"Sentinel QID 1185869 Found: {sentinel_doc is not None}")
    if sentinel_doc:
        print(f"Sentinel QID: {sentinel_doc[1]} | Lang: {sentinel_doc[2]}")
        print(f"Sentinel Query: {sentinel_doc[3].encode('ascii', errors='replace').decode('ascii')}")
    print("=" * 70)

    # Also copy to backend/data/ for local and deployed consistency
    if DB_FILE.exists():
        import shutil
        shutil.copy2(DB_FILE, BACKEND_DB_FILE)
        print(f"Synced database to: {BACKEND_DB_FILE}")


if __name__ == "__main__":
    build_sqlite_database()
