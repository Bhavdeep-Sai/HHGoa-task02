import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from backend.app.config import settings
from backend.app.vector_store.schema import (
    PASSAGES_COLLECTION,
    QA_COLLECTION,
    ChunkMetadata,
    PointRecord
)


class QdrantStore:
    def __init__(self, url: str = None, api_key: str = None):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self._initialized_collections = set()
        
        if self.url and self.url != ":memory:":
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        elif settings.INDEX_MODE == "real" or (hasattr(settings, "QDRANT_STORAGE_PATH") and settings.QDRANT_STORAGE_PATH):
            import os
            os.makedirs(settings.QDRANT_STORAGE_PATH, exist_ok=True)
            try:
                self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
            except RuntimeError as e:
                if "already accessed by another instance" in str(e):
                    lock_file = os.path.join(settings.QDRANT_STORAGE_PATH, ".lock")
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
                        except Exception:
                            raise e
                    else:
                        raise e
                else:
                    raise e
        else:
            self.client = QdrantClient(location=":memory:")

    def init_collections(self, vector_size: int = 384):
        """Initializes collections if they don't already exist."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception:
            existing = []
        
        for coll_name in [PASSAGES_COLLECTION, QA_COLLECTION]:
            if coll_name not in existing:
                try:
                    self.client.create_collection(
                        collection_name=coll_name,
                        vectors_config=qmodels.VectorParams(
                            size=vector_size,
                            distance=qmodels.Distance.COSINE
                        )
                    )
                except Exception:
                    pass
            self._initialized_collections.add(coll_name)

    def upsert_points(self, collection_name: str, records: List[PointRecord]):
        if not records:
            return
        
        points = [
            qmodels.PointStruct(
                id=r.id if r.id else str(uuid.uuid4()),
                vector=r.vector,
                payload=r.payload
            )
            for r in records
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 20,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        is_selected: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        must_filters = []
        if language and language != "en":
            must_filters.append(
                qmodels.FieldCondition(
                    key="language",
                    match=qmodels.MatchValue(value=language)
                )
            )
        if chunk_type:
            must_filters.append(
                qmodels.FieldCondition(
                    key="chunk_type",
                    match=qmodels.MatchValue(value=chunk_type)
                )
            )
        if is_selected is not None:
            must_filters.append(
                qmodels.FieldCondition(
                    key="is_selected",
                    match=qmodels.MatchValue(value=is_selected)
                )
            )

        # Fast path check using in-memory set to avoid per-query RPC call
        if collection_name not in self._initialized_collections:
            self.init_collections()

        query_filter = qmodels.Filter(must=must_filters) if must_filters else None

        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            points = res.points
        elif hasattr(self.client, "search"):
            points = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )
        else:
            points = []

        output = []
        for res_point in points:
            output.append({
                "id": str(res_point.id),
                "score": float(res_point.score),
                "payload": res_point.payload
            })
        return output

