import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    data_dir: str = "data"
    laws_dir: str = os.path.join("law", "laws")

    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "ollama")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")

    chat_provider: str = os.getenv("CHAT_PROVIDER", "ollama")
    chat_model: str = os.getenv("CHAT_MODEL", "llama3.1:8b-instruct-q4_0")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def get_settings() -> Settings:
    load_dotenv(override=False)
    # Re-read after dotenv
    return Settings(
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "ollama"),
        embeddings_model=os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text"),
        chat_provider=os.getenv("CHAT_PROVIDER", "ollama"),
        chat_model=os.getenv("CHAT_MODEL", "llama3.1:8b-instruct-q4_0"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

