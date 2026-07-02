import os
import hashlib
import tiktoken
from typing import List, Dict, Any, Generator

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from config.settings import settings
from src.exceptions import OffTopicException, DocumentIngestionError
from src.logger import get_logger

logger = get_logger(__name__)

class RouterDecision(BaseModel):
    is_on_topic: bool = Field(description="True if relevant to professional QA, False if maliciously off-topic.")
    reason: str = Field(description="Short reason for the classification.")

class LocalRAGEngine:
    def __init__(self, session_id: str = "langchain", history_aware: bool = False):
        self.session_id = session_id
        self.history_aware = history_aware
        self.user_data_dir = os.path.join(settings.data_dir, session_id) if session_id != "langchain" else settings.data_dir
        os.makedirs(self.user_data_dir, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # 1 & 2. Integrate QdrantClient and Connect via Settings
        self.qdrant_client = QdrantClient(url=settings.vector_store_url)
        
        # 3. Multi-Tenant Collection Isolation
        self.ingested_hashes = set()
        has_docs = False
        if not self.qdrant_client.collection_exists(collection_name=self.session_id):
            # all-MiniLM-L6-v2 outputs 384 dimensions
            self.qdrant_client.create_collection(
                collection_name=self.session_id,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        else:
            collection_info = self.qdrant_client.get_collection(self.session_id)
            if collection_info.points_count > 0:
                has_docs = True
            scroll_results, _ = self.qdrant_client.scroll(
                collection_name=self.session_id,
                scroll_filter=None,
                with_payload=["file_hash"],
                limit=10000
            )
            for record in scroll_results:
                if record.payload and "file_hash" in record.payload:
                    self.ingested_hashes.add(record.payload["file_hash"])
            
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
        self.llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=settings.llm_temperature, num_ctx=settings.llm_num_ctx)
        
        self.vectorstore = "NO_DOCS"
        if has_docs:
            self._init_vectorstore()

    def _init_vectorstore(self):
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.session_id,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name="sparse",
            retrieval_mode=RetrievalMode.HYBRID,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retriever_k,
                "fetch_k": settings.retriever_k * 3,
                "lambda_mult": 0.6
            }
        )
        self._setup_lcel_graph()

    def ingest_single_document(self, filename: str, file_bytes: bytes) -> bool:
        """Ingests a single document incrementally if its hash is brand new."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        file_hash = hasher.hexdigest()

        if file_hash in self.ingested_hashes:
            logger.info(f"Skipping {filename}: already ingested (Hash match).")
            return False

        file_path = os.path.join(self.user_data_dir, filename)
        with open(file_path, "wb") as f_out:
            f_out.write(file_bytes)

        logger.info(f"Loading new document: {filename}")
        try:
            if filename.endswith('.txt'):
                loader = TextLoader(file_path)
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith('.md'):
                loader = UnstructuredMarkdownLoader(file_path)
            elif filename.endswith('.html'):
                loader = UnstructuredHTMLLoader(file_path)
            else:
                raise DocumentIngestionError(f"Unsupported file format: {filename}")
            
            docs = loader.load()
            for d in docs:
                d.metadata["file_hash"] = file_hash
                
            text_splitter = SemanticChunker(
                self.embeddings, 
                breakpoint_threshold_type="percentile", 
                breakpoint_threshold_amount=settings.chunk_percentile
            )
            splits = text_splitter.split_documents(docs)
            
            if self.vectorstore == "NO_DOCS":
                self._init_vectorstore()
                
            self.vectorstore.add_documents(documents=splits)
            self.ingested_hashes.add(file_hash)
            return True
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise DocumentIngestionError(f"Failed to load {filename}") from e

    def _setup_lcel_graph(self):
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", settings.unified_router_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
        ]).partial(format_instructions="Return a JSON object strictly matching the schema with 'is_on_topic' (boolean) and 'reason' (string).")
        
        # Use native Ollama structured output support instead of brittle manual parsing
        structured_llm = self.llm.with_structured_output(RouterDecision)
        router_chain = router_prompt | structured_llm
        
        def resolve_query(inputs):
            result = router_chain.invoke(inputs)
            if not result.is_on_topic:
                logger.warning(f"Query rejected. Reason: {result.reason}")
                raise OffTopicException("OFF_TOPIC")
            return inputs["input"]

        def retrieve_and_rerank(query):
            docs = self.retriever.invoke(query)
            if not docs:
                return []
            scores = self.cross_encoder.score([(query, doc.page_content) for doc in docs])
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:settings.reranker_top_n]]
            
            # Simple truncation
            current_tokens = 0
            truncated_docs = []
            max_tokens = settings.max_context_characters // 4
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = tiktoken.get_encoding("gpt2")
            for doc in top_docs:
                doc_tokens = len(enc.encode(doc.page_content))
                if current_tokens + doc_tokens > max_tokens:
                    break
                current_tokens += doc_tokens
                truncated_docs.append(doc)
            return truncated_docs

        self.retrieve_chain = RunnableLambda(resolve_query) | RunnableLambda(retrieve_and_rerank)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def get_qa_prompt(inputs):
            if self.history_aware:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", settings.system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                return prompt_template.invoke({
                    "context": format_docs(inputs["context"]),
                    "input": inputs["input"],
                    "chat_history": inputs.get("chat_history", [])
                })
            else:
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", settings.system_prompt),
                    ("human", "{input}"),
                ])
                return prompt_template.invoke({
                    "context": format_docs(inputs["context"]),
                    "input": inputs["input"]
                })
                
        self.get_qa_prompt = get_qa_prompt

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.vectorstore == "NO_DOCS":
            return {"answer": "NO_DOCS", "context": []}
        context = self.retrieve_chain.invoke(inputs)
        inputs["context"] = context
        prompt_val = self.get_qa_prompt(inputs)
        answer = self.llm.invoke(prompt_val).content
        return {"answer": answer, "context": context}
        
    def stream(self, inputs: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        if self.vectorstore == "NO_DOCS":
            yield {"answer": "NO_DOCS"}
            return
            
        context = self.retrieve_chain.invoke(inputs)
        inputs["context"] = context
        yield {"context": context}
        
        prompt_val = self.get_qa_prompt(inputs)
        for chunk in self.llm.stream(prompt_val):
            yield {"answer": chunk.content}

    def delete_document(self, filename: str) -> bool:
        """Deletes a document from the Qdrant vector database using a metadata filter."""
        file_path = os.path.join(self.user_data_dir, filename)
        try:
            # Drop points where metadata.source exactly matches the file path
            self.qdrant_client.delete(
                collection_name=self.session_id,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=file_path)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Successfully deleted {filename} from collection {self.session_id}.")
            
            # Check if the collection is now empty
            collection_info = self.qdrant_client.get_collection(self.session_id)
            if collection_info.points_count == 0:
                self.vectorstore = "NO_DOCS"
                logger.info(f"Collection {self.session_id} is now empty.")
                
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {filename} from Qdrant: {str(e)}")
            return False
