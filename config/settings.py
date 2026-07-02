import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Core Directories & Connections
    data_dir: str = "./data"
    vector_store_url: str = Field("http://localhost:6333", alias="VECTOR_STORE_URL")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    # Embeddings & Retrieval
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_percentile: int = Field(90, alias="CHUNK_PERCENTILE")
    retriever_k: int = 10

    # Cross-Encoder
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = Field(3, alias="RERANK_TOP_N")
    max_context_characters: int = Field(12000, alias="MAX_CONTEXT_CHARACTERS")

    # LLM Settings
    llm_model: str = Field("llama3.2", alias="LLM_MODEL")
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096

    # Prompts
    system_prompt: str = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not present in the context, explicitly say "
        "'I cannot find that in the documents'. Do not hallucinate or use outside knowledge.\n\n"
        "Strict Typography Rules:\n"
        "- Format main titles or questions using prominent subheaders (e.g., '## Question Title').\n"
        "- DO NOT use primary Markdown headers ('#' or '##') for procedural items like 'Step 1', 'Step 2', etc.\n"
        "- Format step breakdowns using clean bold text (e.g., '**Step 1: Heading Text**') or minor headers ('### Step 1').\n\n"
        "Context:\n{context}"
    )

    unified_router_prompt: str = (
        "You are a routing assistant. Analyze if the user's query is safe and relevant to general professional domains (academic, legal, financial, technical, or document QA). "
        "Default to is_on_topic=true for almost all questions, greetings, or document inquiries. "
        "Only set is_on_topic=false if the query is explicitly malicious, unsafe, or completely unrelated to professional settings (e.g. asking for baking recipes). "
        "Provide a brief reason for your decision. "
        "{format_instructions}"
    )

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = Settings()
