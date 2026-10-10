"""ask_notes: ответ на вопрос инженера строго по документам vault (mdrack + LM Studio)."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, TypeVar

from hayhooks import BasePipelineWrapper
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.component import component
from haystack.dataclasses import ChatMessage, Document
from haystack.utils import Secret
from loguru import logger

from mdrack.config.settings import load_settings
from mdrack.embeddings.lmstudio import LMStudioProvider
from mdrack.search.engine import SearchEngine
from mdrack.storage.sqlite.connection import get_connection
from mdrack.storage.sqlite.fts import FTSIndex
from mdrack.storage.sqlite.repositories import ChunkRepository, FileRepository
from mdrack.storage.sqlite.vector import VectorIndex

LMSTUDIO_URL = "http://localhost:1234/v1"
CHAT_MODEL = "google/gemma-4-e4b"
SEARCH_LIMIT = 6
DOC_MAX_CHARS = 2000

SEARCH_TIMEOUT_S = 40.0
LLM_TIMEOUT_S = 120.0
LLM_DEADLINE_S = LLM_TIMEOUT_S + 15.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0

T = TypeVar("T")

SYSTEM_PROMPT = (
    "Ты инженерный ассистент по базе знаний в Obsidian. "
    "Отвечай ТОЛЬКО на основе предоставленных документов. "
    "Если в документах нет ответа, прямо скажи об этом и не придумывай. "
    "Не подставляй номера стандартов, даты и значения из памяти: "
    "используй только то, что есть в документах. "
    "Ссылайся на источники в формате [chunk_id] прямо в тексте ответа."
)

USER_PROMPT = """Документы:

{% for doc in documents %}
[{{ doc.meta.chunk_id }}] {{ doc.meta.source }}
{{ doc.content }}
{% endfor %}

Вопрос: {{ question }}
"""

FALLBACK_MESSAGE = (
    "Сервис временно недоступен: {reason}. "
    "Попробуйте повторить запрос чуть позже."
)


class ServiceTimeout(RuntimeError):
    """Шаг (поиск или модель) не уложился в отведённое время."""


@component
class MdrackRetriever:
    """Гибридный поиск (FTS5 + вектор, RRF) по глобальному индексу vault."""

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, list[Document]]:
        settings = load_settings(root=Path.cwd(), global_mode=True)
        conn = get_connection(settings.db_path)
        try:
            provider = LMStudioProvider(
                base_url=settings.embedding.base_url,
                model=settings.embedding.model,
                dimensions=settings.embedding.dimensions,
                batch_size=settings.embedding.batch_size,
                timeout_seconds=settings.embedding.timeout_seconds,
                query_instruction=settings.embedding.query_instruction,
            )
            engine = SearchEngine(
                fts=FTSIndex(conn),
                vector_index=VectorIndex(conn),
                embedding_provider=provider,
            )
            hits = engine.search(query, mode="hybrid", limit=SEARCH_LIMIT)

            chunk_repo = ChunkRepository(conn)
            file_repo = FileRepository(conn)
            documents: list[Document] = []
            for chunk_id, score in hits:
                chunk = chunk_repo.get_by_id(chunk_id)
                if not chunk:
                    continue
                file_row = file_repo.get_by_id(chunk["file_id"])
                source = file_row["relative_path"] if file_row else chunk["file_id"]
                documents.append(
                    Document(
                        content=chunk["content"][:DOC_MAX_CHARS],
                        meta={
                            "chunk_id": chunk_id,
                            "source": source,
                            "heading": chunk.get("heading_path_json", ""),
                            "score": round(score, 4),
                        },
                    )
                )
            return {"documents": documents}
        finally:
            conn.close()


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ask_notes")

        self.retriever = MdrackRetriever()
        self.prompt_builder = ChatPromptBuilder(
            template=[
                ChatMessage.from_system(SYSTEM_PROMPT),
                ChatMessage.from_user(USER_PROMPT),
            ],
            required_variables="*",
        )
        self.generator = OpenAIChatGenerator(
            model=CHAT_MODEL,
            api_base_url=LMSTUDIO_URL,
            api_key=Secret.from_token("lm-studio"),
            timeout=LLM_TIMEOUT_S,
            max_retries=0,
            generation_kwargs={"temperature": 0.2, "max_tokens": 1024},
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt", self.prompt_builder)
        self.pipeline.add_component("llm", self.generator)
        self.pipeline.connect("retriever.documents", "prompt.documents")
        self.pipeline.connect("prompt.prompt", "llm.messages")

    def _reset_pool(self) -> None:
        self._pool.shutdown(wait=False)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ask_notes")

    def _call_with_deadline(self, step: str, timeout_s: float, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        future = self._pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            self._reset_pool()
            raise ServiceTimeout(f"{step} не ответил за {timeout_s:g} с") from exc

    def _search(self, question: str) -> list[Document]:
        result = self._call_with_deadline("поиск mdrack", SEARCH_TIMEOUT_S, self.retriever.run, query=question)
        return result["documents"]

    def _generate(self, question: str, documents: list[Document]) -> str:
        rendered = self.prompt_builder.run(question=question, documents=documents)["prompt"]
        result = self._call_with_deadline(
            "модель LM Studio", LLM_DEADLINE_S, self.generator.run, messages=rendered
        )
        replies = result["replies"]
        if not replies or not replies[0].text:
            raise RuntimeError("модель вернула пустой ответ")
        return replies[0].text

    @staticmethod
    def _reason(exc: Exception) -> str:
        text = (str(exc).strip() or type(exc).__name__).rstrip(".")
        return text[:200]

    @staticmethod
    def _with_sources(answer: str, documents: list[Document]) -> str:
        sources = "\n".join(
            f"- {doc.meta['source']} [{doc.meta['chunk_id']}]" for doc in documents
        )
        return f"{answer}\n\nИсточники:\n{sources}"

    def run_api(self, question: str) -> str:
        """
        Ответить на вопрос инженера по базе знаний (ГОСТы, регламенты).
        Только из найденных документов, без домыслов.

        Args:
            question: Вопрос инженера
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                step = time.perf_counter()
                documents = self._search(question)
                search_s = time.perf_counter() - step
                if not documents:
                    logger.info(
                        "ask_notes: совпадений нет за {:.1f} с (попытка {})",
                        time.perf_counter() - started,
                        attempt,
                    )
                    return "В индексе не нашлось документов по этому запросу."

                step = time.perf_counter()
                answer = self._generate(question, documents)
                llm_s = time.perf_counter() - step

                logger.info(
                    "ask_notes: ответ за {:.1f} с (попытка {}, поиск {:.1f} с, "
                    "генерация {:.1f} с, чанков {})",
                    time.perf_counter() - started,
                    attempt,
                    search_s,
                    llm_s,
                    len(documents),
                )
                return self._with_sources(answer, documents)
            except ValueError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = BACKOFF_BASE_S * 2 ** (attempt - 1)
                logger.warning(
                    "ask_notes: попытка {} из {} упала через {:.1f} с ({}: {}), "
                    "повтор через {:.0f} с",
                    attempt,
                    MAX_ATTEMPTS,
                    time.perf_counter() - started,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)

        logger.error(
            "ask_notes: фолбэк после {} попыток за {:.1f} с: {}",
            MAX_ATTEMPTS,
            time.perf_counter() - started,
            last_error,
        )
        reason = self._reason(last_error) if last_error else "нет ответа от сервиса"
        return FALLBACK_MESSAGE.format(reason=reason)
