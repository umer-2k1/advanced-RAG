from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.embeddings import FastEmbedEmbeddings 
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_ollama.llms import OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
from uuid import uuid4
import os
 
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


COLLECTION_NAME = f"multi-query-retrieval-collection"
ollama_model = "mistral:latest"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(
    model=ollama_model,
)

llm = OllamaLLM(model=ollama_model, streaming=True)
 

embeddings = OllamaEmbeddings(
    model=ollama_model,
)



if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
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
    return splitteed_docs

def upload_to_qdrant(documents):
   QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
   
def create_multi_query_retriever(query):
    base_retriever = vector_store.as_retriever()
    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=base_retriever,
        # query_prompt=query_prompt, 
    )
    unique_docs = retriever.invoke(query)
    print("unique_docs::::::", unique_docs)
    print("unique_docs length::::::", len(unique_docs))
    return unique_docs
    
     

# def query_with_multi_query_retriever(query, k_initial=15):
#     retriever = create_multi_query_retriever(query, k_initial)
#     results = retriever.get_relevant_documents(query)
#     context = "\n\n".join([doc.page_content for doc in results])
#     answer = answer_chain.invoke({"query": query, "context": context})
#     return answer["text"]

if __name__ == "__main__":
    pdf_path = "./sample/sql.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("DOCS:::::", docs)
    splitted_docs = split_documents(docs)
    upload_to_qdrant(splitted_docs)
    
    while True:
        print("\n\n ----------------------------------")    
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")
        
        question = input("Ask a question about the video (or type 'q' to quit):\n")
        if question.lower() == "q":
            break
        response = create_multi_query_retriever(question)
        print("🔹 Answering...")            
        print("\nAnswer:\n", response)