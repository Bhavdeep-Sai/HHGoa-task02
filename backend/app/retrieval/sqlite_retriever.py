import os
import re
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend.app.config import settings
from backend.app.utils.logger import logger


class SQLiteRetriever:
    """
    High-Performance, Zero-RAM-Overhead SQLite FTS5 Multilingual Lexical Retriever.
    Queries read-only SQLite FTS5 table with BM25 ranking in < 2ms,
    returning standard RAG evidence candidates with complete QID provenance.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = self._resolve_db_path(db_path)
        self._verify_db()

    @staticmethod
    def _resolve_db_path(custom_path: Optional[str] = None) -> Path:
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                return p.resolve()

        configured_path = Path(getattr(settings, "SQLITE_STORAGE_PATH", "./data/msmarco_xi.sqlite"))
        if configured_path.exists():
            return configured_path.resolve()

        this_file = Path(__file__).resolve()
        backend_dir = this_file.parent.parent.parent  # backend/app/retrieval -> backend
        project_root = backend_dir.parent

        candidates = [
            backend_dir / "data" / "msmarco_xi.sqlite",
            project_root / "data" / "msmarco_xi.sqlite",
            Path.cwd() / "data" / "msmarco_xi.sqlite",
            Path.cwd().parent / "data" / "msmarco_xi.sqlite",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()

        return configured_path.resolve()

    def _get_connection(self) -> sqlite3.Connection:
        """Opens a lightweight read-only connection to the SQLite database."""
        db_str = str(self.db_path)
        try:
            # Use URI mode for strict read-only access
            uri = f"file:{db_str}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except Exception:
            conn = sqlite3.connect(db_str, check_same_thread=False)

        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only = ON;")
        except Exception:
            pass
        return conn

    def _verify_db(self):
        """Verifies database connectivity and FTS5 table at initialization."""
        if not self.db_path.exists():
            logger.warning(f"[SQLiteRetriever] Database file does not exist at '{self.db_path}'.")
            return

        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents_fts;")
            fts_count = cur.fetchone()[0]
            conn.close()
            logger.info(
                f"[SQLiteRetriever] Ready. Database: '{self.db_path}' ({doc_count} docs, {fts_count} FTS rows)."
            )
        except Exception as e:
            logger.error(f"[SQLiteRetriever] Error verifying database '{self.db_path}': {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Safe diagnostics for health check and monitoring without exposing secrets."""
        exists = self.db_path.exists()
        size_mb = round(self.db_path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
        doc_count = 0
        fts_exists = False
        sentinel_found = False

        if exists:
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM documents;")
                doc_count = cur.fetchone()[0]
                cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents_fts';")
                fts_exists = cur.fetchone() is not None
                cur.execute("SELECT 1 FROM documents WHERE qid = '1185869' LIMIT 1;")
                sentinel_found = cur.fetchone() is not None
                conn.close()
            except Exception as e:
                logger.warning(f"[SQLiteRetriever] Diagnostics query error: {e}")

        return {
            "database_path": str(self.db_path.name),
            "database_exists": exists,
            "database_size_mb": size_mb,
            "document_count": doc_count,
            "fts_table_exists": fts_exists,
            "sentinel_query_id": "1185869",
            "sentinel_found": sentinel_found,
            "retrieval_mode": "sqlite",
            "retrieval_ready": exists and doc_count > 0 and fts_exists
        }

    # Stop words that should not form mandatory FTS query constraints
    STOP_WORDS = {
        "what", "was", "is", "are", "were", "the", "a", "an", "and", "or", "of", "to", "in", "for",
        "on", "at", "by", "with", "from", "it", "its", "did", "does", "do", "how", "why", "when",
        "where", "who", "which", "kya", "hua", "hone", "ke", "ka", "ki", "baad", "mein", "ko",
        "tha", "thi", "hai", "ho", "aaj", "kal", "par", "is", "se", "ne"
    }

    def normalize_and_build_fts_query(self, query: str) -> str:
        """
        Extracts content tokens, handles multilingual Unicode characters safely,
        and constructs an FTS5 search expression with phrase boosting and token expansion.
        """
        raw_tokens = re.findall(r'[\w]+', query.lower())
        if not raw_tokens:
            return ""

        content_tokens = [
            t for t in raw_tokens
            if (len(t) > 1 or ord(t[0]) > 127) and t not in self.STOP_WORDS
        ]

        if not content_tokens:
            content_tokens = [t for t in raw_tokens if len(t) > 1 or ord(t[0]) > 127]

        if not content_tokens:
            return ""

        # Escape tokens for FTS5 (double quote each token)
        safe_tokens = [f'"{t}"' for t in content_tokens]

        # Construct query:
        # Exact phrase query (high precision) OR individual tokens (high recall)
        if len(safe_tokens) > 1:
            phrase_str = f'"{" ".join(content_tokens)}"'
            or_str = " OR ".join(safe_tokens)
            return f"{phrase_str} OR ({or_str})"
        else:
            return safe_tokens[0]

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches SQLite FTS5 database and returns top K evidence candidates.
        """
        if not self.db_path.exists():
            return []

        fts_query = self.normalize_and_build_fts_query(query)
        if not fts_query:
            return []

        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Retrieve top candidates ranked by FTS5 BM25
            sql = """
            SELECT 
                d.id,
                d.qid,
                d.language,
                d.query,
                d.answer,
                d.passage,
                d.source,
                d.metadata_json,
                bm25(documents_fts, 5.0, 3.0, 1.0) AS fts_score
            FROM documents d
            JOIN documents_fts ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY fts_score ASC
            LIMIT ?;
            """
            cur.execute(sql, (fts_query, top_k * 3))
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"[SQLiteRetriever] Search error for query '{query}': {e}")
            return []

        if not rows:
            return []

        candidates = []
        seen_parents = set()

        for r in rows:
            doc_id = r["id"]
            qid = r["qid"]
            lang = r["language"] or "en"
            q_text = r["query"] or ""
            a_text = r["answer"] or ""
            p_text = r["passage"] or ""
            source = r["source"] or "ai4bharat/MSMARCO-XI"
            raw_meta = r["metadata_json"] or "{}"

            try:
                meta = json.loads(raw_meta)
            except Exception:
                meta = {}

            parent_id = meta.get("parent_id") or f"{qid}_p0"
            doc_key = f"{qid}_{parent_id}"
            if doc_key in seen_parents:
                continue
            seen_parents.add(doc_key)

            # Convert negative SQLite BM25 score to positive relevance score (~0..50)
            raw_bm25 = abs(float(r["fts_score"]))
            scaled_score = min(50.0, max(5.0, raw_bm25))

            chunk_id = str(meta.get("chunk_id") or f"doc_{doc_id}")
            chunk_type = meta.get("chunk_type", "passage")
            is_qa = chunk_type == "qa" or (a_text and not p_text)

            candidate = {
                "id": chunk_id,
                "score": round(scaled_score, 4),
                "document_type": "qa_unit" if is_qa else "passage",
                "retrieval_method": "sqlite_fts",
                "dense_score": None,
                "bm25_score": round(scaled_score, 4),
                "rrf_score": 0.016,
                "reranker_score": 0.0,
                "text": p_text or a_text or q_text,
                "payload": {
                    "query_id": int(qid) if qid.isdigit() else qid,
                    "parent_id": parent_id,
                    "passage_id": meta.get("passage_id", 0),
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "language": lang,
                    "dataset": source,
                    "split": meta.get("split", "train"),
                    "query": q_text,
                    "query_en": meta.get("query_en", q_text),
                    "query_hi": meta.get("query_hi", q_text),
                    "answer": a_text,
                    "answer_hi": meta.get("answer_hi", a_text),
                    "text": p_text or a_text,
                    "parent_text": p_text,
                    "is_selected": meta.get("is_selected", True)
                }
            }

            # If language filtering requested and not English, prioritize matching language
            if language and language != "en" and lang != language and lang != "en":
                # Deprioritize mismatching third language
                candidate["score"] *= 0.5

            candidates.append(candidate)
            if len(candidates) >= top_k:
                break

        # Sort by final score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


_sqlite_retriever_instance: Optional[SQLiteRetriever] = None


def get_sqlite_retriever() -> SQLiteRetriever:
    global _sqlite_retriever_instance
    if _sqlite_retriever_instance is None:
        _sqlite_retriever_instance = SQLiteRetriever()
    return _sqlite_retriever_instance
