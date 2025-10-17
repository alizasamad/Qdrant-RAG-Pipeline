from typing import List, Union, Generator, Iterator, Sequence, Any
from pydantic import BaseModel, Field
from llama_index.core import PromptTemplate, VectorStoreIndex, StorageContext
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse, ChatMessage, ChatResponse
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from qdrant_client import QdrantClient
import os
import requests
import logging
log = logging.getLogger(__name__)

class OpenWebUILLM(LLM):
    # Declare Pydantic fields
    api_key: str = Field(default = None, description = "API key for OpenWebUI")
    model: str = Field(default = "llama3.2:latest", description = "Model name")
    max_tokens: int = Field(default=1000, description = "Maximum tokens to generate")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    owui_host: str = Field(default="host.docker.internal", description = "OWUI Host")
    owui_port: int = Field(default=3000, description = "OWUI Port")
    

    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float, owui_host: str, owui_port: int, **kwargs):
        # Initialize with field values
        super().__init__(
            api_key = api_key,
            model = model,
            max_tokens = max_tokens,
            temperature = temperature,
            owui_host = owui_host,
            owui_port = owui_port,
            **kwargs
        )
    
    @property
    def metadata(self):
        return {'model': self.model}
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        log.info(f"actual api-key: {self.api_key}")
        try:
            response = requests.post(
                url = f"http://{self.owui_host}:{self.owui_port}/api/chat/completions",
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                },
                timeout = 600
            )

            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                return CompletionResponse(text=text)
            else:
                return CompletionResponse(text=f"API Error {response.status_code}: {response.text}")
            
        except Exception as e:
            return CompletionResponse(text=f"Error calling LLM: {str(e)}")
        
    def stream_complete(self, prompt: str, **kwargs):
        completion = self.complete(prompt, **kwargs)
        yield completion

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        prompt_parts  = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            prompt_parts.append(f"{role}: {msg.content}")

        prompt = "\n".join(prompt_parts)
        completion = self.complete(prompt, **kwargs)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=completion.text)
        )
    
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs):
        chat_response = self.chat(messages, **kwargs)
        yield chat_response
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs):
        completion = await self.acomplete(prompt, **kwargs)
        yield completion

    async def achat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs):
        chat_response = await self.achat(messages, **kwargs)
        yield chat_response

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""
    llm: OpenWebUILLM
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str, nodes):
        log.info(f"User query: {query_str}")

        # Create formatted content with citations
        context_parts = []
        sources = []

        for i, node_with_score in enumerate(nodes):
            # Get content
            content = node_with_score.node.get_content()

            # Get source information
            inner_metadata = node_with_score.node.metadata.get('metadata', {})
            source = inner_metadata.get('source', 'Unknown source')

            # Add to context with citation marker
            context_parts.append(f"[Excerpt {i+1}]: {content}")

            # Add to sources
            if source not in [s['source'] for s in sources]:
                sources.append({
                    'source': source,
                    'id': i+1
                })
            
        # Format context with all content
        context_str = "\n\n".join(context_parts)
        log.info(f'''Context \n\n
                    {context_str}''')
        
        # Format sources for citations
        sources_str = "\n".join([f"[{s['id']}] {s['source']}" for s in sources])

        # Build a complete prompt with citations section
        complete_prompt = self.qa_prompt.format(
            context_str = context_str,
            query_str = query_str,
            sources_str = sources_str
        )

        # Use LLM to generate response
        log.info("Sending full context to LLM ...")
        response = self.llm.complete(complete_prompt)
        log.info(f'''LLM response generated: \n\n 
                    {response} ''')
        return str(response)
    
class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_API_KEY: str
        QDRANT_COLLECTION_NAME: str
        OWUI_HOST: str
        OWUI_PORT: int
        OWUI_API_KEY: str
        EMBED_MODEL: str
        LANG_MODEL: str

    def __init__(self):
        self.name = "Qdrant RAG Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "host.docker.internal"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "my-secure-readonly-key-456"),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "open-webui_files"),
                "OWUI_HOST": os.getenv("OWUI_HOST", "host.docker.internal"),
                "OWUI_PORT": int(os.getenv("OWUI_PORT", "3000")),
                "OWUI_API_KEY": os.getenv("OWUI_API_KEY", "<insert-key-here>"),
                "EMBED_MODEL": os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                "LANG_MODEL": os.getenv("LANG_MODEL", "llama3.2:latest")
            }
        )

        # Configure embedding model
        embed_model = HuggingFaceEmbedding(model_name = self.valves.EMBED_MODEL)
        Settings.embed_model = embed_model

        # Connect to existing Qdrant collection
        client = QdrantClient(url=f"http://{self.valves.QDRANT_HOST}:{self.valves.QDRANT_PORT}",
                api_key = self.valves.QDRANT_API_KEY
        )

        # Create a vector store from existing collection
        vector_store = QdrantVectorStore(
            client=client,
            collection_name = self.valves.QDRANT_COLLECTION_NAME,
            vector_name = ""  # Use default vector field for OWUI
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from existing vector store (not from documents)
        index = VectorStoreIndex.from_vector_store(
            vector_store = vector_store,
            storage_context = storage_context
        )

        # Create retriever directly from vector store
        self.retriever = index.as_retriever(similarity_top_k = 3)

        # Define the prompt template
        self.qa_prompt = PromptTemplate(
            "Context information is below. \n"
            "-------------------------\n"
            "{context_str}\n"
            "-------------------------\n"
            "Sources:\n"
            "{sources_str}\n"
            "-------------------------\n"
            "You are a helpful AI Assistant providing information from a knowledge base.\n"
            "Generate human readable output based only on the provided context. \n"
            "Use the same language as the user's query.\n"
            "IMPORTANT: Cite sources using their numbers when answering. \n"
            "Given the context information and not prior knowledge, answer the query. \n"
            "Query: {query_str}\n"
            "Answer: "
        )
    
    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            log.info("attempting to connect to LLM...")
            log.info(f"api-key: {self.valves.OWUI_API_KEY}")
            llm = OpenWebUILLM(
                api_key = self.valves.OWUI_API_KEY,
                owui_host = self.valves.OWUI_HOST,
                owui_port = self.valves.OWUI_PORT,
                model = self.valves.LANG_MODEL,
                max_tokens = 1000,
                temperature = 0.1,
            )

            log.info("attempting to perform RAG search ...")
            query_engine = RAGStringQueryEngine(
                llm=llm,
                qa_prompt = self.qa_prompt,
            )

            # Step 1: Retrieve nodes
            nodes = self.retriever.retrieve(user_message)
            log.info(f'''Here are the nodes:\n\n
                     {nodes} ''')
            
            # Step 2: Emit citations for each node
            for i, node_with_score in enumerate(nodes):
                node = node_with_score.node
                inner_metadata = node.metadata.get("metadata", {})
                source_name = inner_metadata.get("source", "Unknown source")

                yield {
                    "event": {
                        "type": "citation",
                        "data": {
                            "document": [node.get_content()],
                            "source": {"name": f"{source_name}",
                                       "url": f"http://localhost:{self.valves.OWUI_PORT}/api/v1/files/{inner_metadata.get('file_id')}/content"
                            },
                            "distances": [float(getattr(node_with_score, "score", 0.0))], 
                            "id": i + 1,
                        },
                    }
                }

            # Step 3: Generate the final answer with citations inline
            response = query_engine.custom_query(user_message, nodes)
            yield response
        
        except Exception as e:
            yield f"Error in RAG pipeline: {str(e)}"