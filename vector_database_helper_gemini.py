import logging
import os
from typing import Any, Literal
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel
from gemini_helper import ask_gemini, text_to_vector

load_dotenv()

logger = logging.getLogger("vector_db_helper")

SearchMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


class RerankedList(BaseModel):
    ordered_ids: list[int]


def get_pg_connection(db_url: str | None = None, register_vec: bool = True) -> psycopg.Connection:
    """
    Opens a PostgreSQL connection with optional pgvector type registration.
    Set register_vec=False before 'CREATE EXTENSION vector' has been run on a fresh DB.
    """
    target_url = db_url or os.getenv("DATABASE_URL")
    if target_url:
        conn = psycopg.connect(target_url)
    else:
        conn = psycopg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "consensus_db"),
            user=os.getenv("POSTGRES_USER", "dba_admin"),
            password=os.getenv("POSTGRES_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE", "prefer"),
        )
    if register_vec:
        register_vector(conn)
    return conn


def init_vector_table(
    table_name: str = "vector_objects",
    dimensions: int = 768,
    db_url: str | None = None,
) -> None:
    """Enables pgvector, creates the storage table, and builds HNSW + GIN indexes."""
    with get_pg_connection(db_url, register_vec=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            register_vector(conn)

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding VECTOR({dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_hnsw
                ON {table_name} USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_fts_gin
                ON {table_name} USING gin (to_tsvector('simple', title || ' ' || content));
                """
            )
        conn.commit()
    logger.info(f"Initialized '{table_name}' (dims={dimensions}) with HNSW + GIN indexes.")


def store_vector_object(
    title: str,
    content: str,
    category: str = "general",
    metadata: dict[str, Any] | None = None,
    table_name: str = "vector_objects",
    dimensions: int = 768,
    db_url: str | None = None,
) -> int:
    """Vectorizes content and stores the record + metadata in PostgreSQL, returning its new ID."""
    vector = text_to_vector(content, dimensions=dimensions)
    meta_payload = Jsonb(metadata or {})

    with get_pg_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table_name} (title, category, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (title, category, content, meta_payload, vector),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve inserted record ID.")
            new_id = int(row[0])
        conn.commit()

    logger.info(f"Stored vector object id={new_id} title='{title}' in '{table_name}'.")
    return new_id


def get_vector_object(
    object_id: int,
    table_name: str = "vector_objects",
    db_url: str | None = None,
) -> dict[str, Any] | None:
    """Retrieves a single stored vector object and its metadata by primary key ID."""
    with get_pg_connection(db_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, title, category, content, metadata, embedding, created_at
                FROM {table_name}
                WHERE id = %s;
                """,
                (object_id,),
            )
            return cur.fetchone()


def search_vector_objects(
    query_text: str,
    search_type: SearchMode = "hybrid",
    top_k: int = 3,
    alpha: float = 0.5,
    language: str = "simple",
    category_filter: str | None = None,
    table_name: str = "vector_objects",
    dimensions: int = 768,
    db_url: str | None = None,
) -> list[dict[str, Any]]:
    """
    Unified retrieval interface supporting 4 production search modes:
      - 'dense': Pure pgvector semantic cosine similarity (<=>)
      - 'sparse': Pure PostgreSQL full-text keyword search (ts_rank_cd + GIN)
      - 'hybrid': Weighted blend of sparse keyword + dense vector scores via alpha
      - 'hybrid_rerank': Two-stage retrieval (Hybrid candidate fetch -> Gemini LLM reranking)
    """
    if search_type not in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        raise ValueError(f"Unsupported search_type: {search_type}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    where_sql = "WHERE category = %s" if category_filter else ""
    filter_params: list[Any] = [category_filter] if category_filter else []
    fetch_limit = top_k * 3 if search_type == "hybrid_rerank" else top_k

    with get_pg_connection(db_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if search_type == "dense":
                query_vector = text_to_vector(query_text, dimensions=dimensions)
                sql = f"""
                    SELECT
                        id, title, category, content, metadata,
                        ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS score
                    FROM {table_name}
                    {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                params = [query_vector, *filter_params, query_vector, fetch_limit]
                cur.execute(sql, params)
                results = cur.fetchall()

            elif search_type == "sparse":
                sql = f"""
                    SELECT
                        id, title, category, content, metadata,
                        ROUND(
                            ts_rank_cd(
                                to_tsvector(%s, title || ' ' || content),
                                plainto_tsquery(%s, %s)
                            )::numeric, 4
                        ) AS score
                    FROM {table_name}
                    {where_sql}
                    ORDER BY score DESC
                    LIMIT %s;
                """
                params = [language, language, query_text, *filter_params, fetch_limit]
                cur.execute(sql, params)
                results = cur.fetchall()

            else:  # 'hybrid' or 'hybrid_rerank'
                query_vector = text_to_vector(query_text, dimensions=dimensions)
                sql = f"""
                    WITH scored AS (
                        SELECT
                            id, title, category, content, metadata,
                            ts_rank_cd(
                                to_tsvector(%s, title || ' ' || content),
                                plainto_tsquery(%s, %s)
                            ) AS sparse_score,
                            (1 - (embedding <=> %s::vector)) AS dense_score
                        FROM {table_name}
                        {where_sql}
                    )
                    SELECT
                        id, title, category, content, metadata,
                        ROUND(sparse_score::numeric, 4) AS sparse_score,
                        ROUND(dense_score::numeric, 4) AS dense_score,
                        ROUND(((sparse_score * %s) + (dense_score * %s))::numeric, 4) AS score
                    FROM scored
                    ORDER BY score DESC
                    LIMIT %s;
                """
                params = [
                    language,
                    language,
                    query_text,
                    query_vector,
                    *filter_params,
                    1.0 - alpha,
                    alpha,
                    fetch_limit,
                ]
                cur.execute(sql, params)
                results = cur.fetchall()

    if search_type == "hybrid_rerank" and len(results) > 1:
        candidates_text = "\n".join(
            f"ID={r['id']} | Title: {r['title']} | Desc: {r['content']} | Meta: {r['metadata']}"
            for r in results
        )
        rerank_prompt = (
            f"User Request: '{query_text}'\n\n"
            f"Rank these candidate items from best match to worst match based strictly on how well "
            f"they satisfy the technical symptoms and constraints:\n{candidates_text}"
        )
        reranked: RerankedList = ask_gemini(
            prompt=rerank_prompt,
            system_instruction="You are a precise search relevance reranker. Return IDs ordered best-first.",
            temperature=0.0,
            response_schema=RerankedList,
        )
        id_map = {r["id"]: r for r in results}
        ordered = [id_map[i] for i in reranked.ordered_ids if i in id_map]
        remaining = [r for r in results if r["id"] not in set(reranked.ordered_ids)]
        results = (ordered + remaining)[:top_k]

    logger.info(f"search_type={search_type} | returned={len(results)} | query='{query_text[:35]}...'")
    return results
