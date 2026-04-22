import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from elasticsearch import AsyncElasticsearch


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_query(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return lowered.strip()


class ConversationMemoryStore:
    def __init__(self) -> None:
        self.enabled = _as_bool(os.getenv("SUPERVISOR_MEMORY_ENABLED"), True)
        self.index_name = os.getenv("SUPERVISOR_MEMORY_INDEX", "ragsapp_conversation_memory")
        self.ttl_seconds = int(os.getenv("SUPERVISOR_MEMORY_TTL_SECONDS", "3600"))
        self.max_answer_chars = int(os.getenv("SUPERVISOR_MEMORY_MAX_ANSWER_CHARS", "4000"))
        self.context_enabled = _as_bool(os.getenv("SUPERVISOR_CONTEXT_ENABLED"), True)
        self.context_max_turns = int(os.getenv("SUPERVISOR_CONTEXT_MAX_TURNS", "8"))
        self.context_timezone = os.getenv("SUPERVISOR_CONTEXT_TIMEZONE", "Asia/Kolkata")
        self.context_day_mode = (os.getenv("SUPERVISOR_CONTEXT_DAY_MODE", "today") or "today").strip().lower()

        self.es_url = os.getenv("RAG_UPLOAD_ELASTIC_URL")
        self.es_api_key = os.getenv("ELASTICSEARCH_API_KEY")
        self._client: Optional[AsyncElasticsearch] = None

    async def startup(self) -> None:
        if not self.enabled:
            return
        if not self.es_url:
            print("Conversation memory disabled: RAG_UPLOAD_ELASTIC_URL is not set.")
            self.enabled = False
            return

        try:
            kwargs: Dict[str, Any] = {
                "request_timeout": 20,
                "retry_on_timeout": True,
            }
            if self.es_api_key:
                kwargs["api_key"] = self.es_api_key
            self._client = AsyncElasticsearch(self.es_url, **kwargs)
            if not await self._client.ping():
                print("Conversation memory disabled: could not ping Elasticsearch.")
                self.enabled = False
                await self.shutdown()
                return
            await self._ensure_index()
        except Exception as exc:
            print(f"Conversation memory startup failed: {exc}")
            self.enabled = False
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._client and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception as exc:
                print(f"Conversation memory client close failed: {exc}")
        self._client = None

    async def get_cached_answer(self, provider: str, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self._client:
            return None

        normalized_query = normalize_query(query)
        if not normalized_query:
            return None

        query_hash = self._query_hash(normalized_query)
        now = datetime.now(timezone.utc)
        from_ts = now - timedelta(seconds=self.ttl_seconds)

        body = {
            "size": 1,
            "sort": [{"created_at": {"order": "desc"}}],
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"provider": provider}},
                        {"term": {"user_id": user_id}},
                        {"term": {"query_hash": query_hash}},
                        {"range": {"created_at": {"gte": from_ts.isoformat()}}},
                    ]
                }
            },
        }

        try:
            response = await self._client.search(index=self.index_name, body=body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return None
            source = hits[0].get("_source", {})
            answer = (source.get("final_answer") or "").strip()
            if not answer:
                return None
            return source
        except Exception as exc:
            print(f"Conversation memory read failed: {exc}")
            return None

    async def save_answer(
        self,
        provider: str,
        user_id: str,
        user_query: str,
        final_answer: str,
        route_type: str,
        reason_code: str,
        confidence: float,
        used_rag: bool,
    ) -> None:
        if not self.enabled or not self._client:
            return

        normalized_query = normalize_query(user_query)
        if not normalized_query:
            return

        answer = (final_answer or "").strip()
        if not answer:
            return

        answer = answer[: self.max_answer_chars]
        doc = {
            "provider": provider,
            "user_id": user_id,
            "user_query": user_query,
            "normalized_query": normalized_query,
            "query_hash": self._query_hash(normalized_query),
            "final_answer": answer,
            "route_type": route_type,
            "reason_code": reason_code,
            "confidence": float(confidence),
            "used_rag": bool(used_rag),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            await self._client.index(index=self.index_name, document=doc)
        except Exception as exc:
            print(f"Conversation memory write failed: {exc}")

    async def get_recent_conversation_history(
        self,
        provider: str,
        user_id: str,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        if not self.enabled or not self.context_enabled or not self._client:
            return []

        turns_limit = max(2, min(int(max_turns or self.context_max_turns), 30))
        doc_limit = max(2, min(turns_limit, 50))
        range_filter = self._build_context_time_filter()

        body = {
            "size": doc_limit,
            "sort": [{"created_at": {"order": "desc"}}],
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"provider": provider}},
                        {"term": {"user_id": user_id}},
                        range_filter,
                    ]
                }
            },
        }

        try:
            response = await self._client.search(index=self.index_name, body=body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return []

            turns: List[Dict[str, str]] = []
            for hit in reversed(hits):
                source = hit.get("_source", {})
                created_at = str(source.get("created_at", ""))
                route_type = str(source.get("route_type", ""))
                reason_code = str(source.get("reason_code", ""))

                user_query = str(source.get("user_query", "")).strip()
                if user_query:
                    turns.append(
                        {
                            "role": "user",
                            "content": user_query,
                            "created_at": created_at,
                            "route_type": route_type,
                            "reason_code": reason_code,
                        }
                    )

                assistant_answer = str(source.get("final_answer", "")).strip()
                if assistant_answer:
                    turns.append(
                        {
                            "role": "assistant",
                            "content": assistant_answer,
                            "created_at": created_at,
                            "route_type": route_type,
                            "reason_code": reason_code,
                        }
                    )

            return turns[-turns_limit:]
        except Exception as exc:
            print(f"Conversation history read failed: {exc}")
            return []

    async def _ensure_index(self) -> None:
        assert self._client is not None
        exists = await self._client.indices.exists(index=self.index_name)
        if exists:
            return

        mapping = {
            "mappings": {
                "properties": {
                    "provider": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "user_query": {"type": "text"},
                    "normalized_query": {"type": "keyword"},
                    "query_hash": {"type": "keyword"},
                    "final_answer": {"type": "text"},
                    "route_type": {"type": "keyword"},
                    "reason_code": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "used_rag": {"type": "boolean"},
                    "created_at": {"type": "date"},
                }
            }
        }
        await self._client.indices.create(index=self.index_name, body=mapping)

    def _build_context_time_filter(self) -> Dict[str, Any]:
        if self.context_day_mode == "today":
            tzinfo = self._resolve_timezone(self.context_timezone)
            now_local = datetime.now(tz=tzinfo)
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            start_utc = start_local.astimezone(timezone.utc)
            return {"range": {"created_at": {"gte": start_utc.isoformat()}}}

        now = datetime.now(timezone.utc)
        from_ts = now - timedelta(seconds=self.ttl_seconds)
        return {"range": {"created_at": {"gte": from_ts.isoformat()}}}

    @staticmethod
    def _resolve_timezone(tz_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            return ZoneInfo("UTC")

    @staticmethod
    def _query_hash(normalized_query: str) -> str:
        return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
