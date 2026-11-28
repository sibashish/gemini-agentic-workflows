import logging
import os
import time
from typing import Type, TypeVar
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

# Route all telemetry and SDK HTTP logs to a dedicated log file instead of stdout
LOG_FILE = os.getenv("GEMINI_LOG_FILE", "gemini_telemetry.log")

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)

# Attach FileHandler to root logger and prevent console spam
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()
root_logger.addHandler(file_handler)

logger = logging.getLogger("gemini_helper")

# 15-second timeout (15,000 ms) so stalled requests fail fast and trigger fallback
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=15_000),
)

DEFAULT_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

DEFAULT_EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "text-embedding-004",
]

DEFAULT_SYSTEM_ROLE = "You are a concise, accurate technical and analytical AI assistant."

T = TypeVar("T", bound=BaseModel)


def format_prompt(template: str, **kwargs) -> str:
    """Safely injects dynamic runtime variables into a prompt template."""
    return template.format(**kwargs)


def ask_gemini(
    prompt: str,
    system_instruction: str = DEFAULT_SYSTEM_ROLE,
    temperature: float = 0.2,
    model: str | None = None,
    response_schema: Type[T] | None = None,
) -> str | T:
    """
    Sends a dynamic prompt variable to Gemini with:
    - Automatic model fallback & timeout protection
    - Optional Pydantic structured JSON output (response_schema)
    - File-based latency and token usage telemetry logging
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt variable cannot be empty.")

    models_to_try = [model] if model else DEFAULT_FALLBACK_MODELS

    config_kwargs = {
        "system_instruction": system_instruction,
        "temperature": temperature,
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    for idx, current_model in enumerate(models_to_try):
        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            usage = getattr(response, "usage_metadata", None)
            in_tokens = getattr(usage, "prompt_token_count", "N/A")
            out_tokens = getattr(usage, "candidates_token_count", "N/A")

            logger.info(
                f"model={current_model} | latency={elapsed_ms:.0f}ms | "
                f"tokens_in={in_tokens} | tokens_out={out_tokens}"
            )

            if response_schema is not None:
                return response_schema.model_validate_json(response.text)

            return response.text

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if idx == len(models_to_try) - 1:
                logger.error(f"All fallback models failed. Last error on {current_model}: {e}")
                raise

            next_model = models_to_try[idx + 1]
            logger.warning(
                f"model={current_model} failed after {elapsed_ms:.0f}ms "
                f"({type(e).__name__}). Falling back to {next_model}..."
            )
            time.sleep(1)


def print_gemini(
    prompt: str,
    system_instruction: str = DEFAULT_SYSTEM_ROLE,
    temperature: float = 0.2,
    model: str | None = None,
) -> None:
    """Executes ask_gemini() for text output and prints directly to the console."""
    result = ask_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        model=model,
    )
    print(result)

# The following is for converting text to vector
def text_to_vector(
    text: str,
    dimensions: int = 768,
    task_type: str = "SEMANTIC_SIMILARITY",
    model: str | None = None,
) -> list[float]:
    """Converts text into a numerical vector embedding with timeout and model fallback."""
    if not text or not text.strip():
        raise ValueError("Input text for vectorization cannot be empty.")

    models_to_try = [model] if model else DEFAULT_EMBEDDING_MODELS

    for idx, current_model in enumerate(models_to_try):
        start_time = time.perf_counter()
        try:
            response = client.models.embed_content(
                model=current_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=dimensions,
                ),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            vector = response.embeddings[0].values

            logger.info(
                f"embed_model={current_model} | dims={len(vector)} | latency={elapsed_ms:.0f}ms"
            )
            return vector

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if idx == len(models_to_try) - 1:
                logger.error(f"All embedding models failed. Last error on {current_model}: {e}")
                raise

            next_model = models_to_try[idx + 1]
            logger.warning(
                f"embed_model={current_model} failed after {elapsed_ms:.0f}ms "
                f"({type(e).__name__}). Falling back to {next_model}..."
            )
            time.sleep(1)

