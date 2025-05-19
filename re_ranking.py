"""
Reranking (Using Cross-Encoders)
What: After retrieving top-k, rerank them using a model that scores relevance in context of the query.
Why: Improves quality of selected passages.
"""

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_ollama.llms import OllamaLLM
from uuid import uuid4
import os

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


COLLECTION_NAME = f"re-ranking-collection"
ollama_model = "mistral:latest"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(
    model=ollama_model,
)

llm = OllamaLLM(model=ollama_model, streaming=True)


if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    
# Initialize vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    splitteed_docs  = splitter.split_documents(documents)
    print("splitters....", splitteed_docs)
    vectordb = QdrantVectorStore.from_documents(
        documents=splitteed_docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    return vectordb

 
 

 
answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question:

Context:
{context}

Question:
{query}

Answer:
""")
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

 


def query_with_flashrank_compression(query, k_initial=15):
    retriever = vector_store.as_retriever(search_kwargs={"k": k_initial})
    
    # Compression Retriever with FlashRank
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

    compressed_docs = compression_retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in compressed_docs])
    print("context::::::", context)
    answer = answer_chain.invoke({"query": query, "context": context})
    return answer["text"]
 

if __name__ == "__main__":
    pdf_path = "./sample/sql.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("DOCS:::::", docs)
    split_documents(docs)
    
    
    while True:
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")
        question = input("Ask a question about the video (or type 'q' to quit):\n")
        if question.lower() == "q":
            break 
        response = query_with_flashrank_compression(question)
        print("🔹 Answering...")
        print("\nAnswer:\n", response)
   