from langchain_qdrant import QdrantVectorStore
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
from dotenv import load_dotenv
import os
 
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


COLLECTION_NAME = f"hybrid-collection"
ollama_model = "mistral:latest"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(
    model=ollama_model,
)

llm = OllamaLLM(model=ollama_model, streaming=True)

# FastEmbed setup (dense + sparse)
embedding = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


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
    return splitteed_docs

def upload_to_qdrant(documents):
   QdrantVectorStore.from_documents(
        documents=splitteed_docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
   
def hybrid_search(query: str,k=5 ):
    vectorstore  = QdrantVectorStore(
         client=client, 
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embedding=embeddings,
        retrieval_mode="hybrid",  #  enables sparse + dense retrieval
        sparse_encoder = FastEmbedSparse(),

    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results  = retriever.get_relevant_documents(query)
    return results 


prompt_template = ChatPromptTemplate.from_template(
    """
    You are an intelligent assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

def query_llm(query, context):
    results : list[Document] = hybrid_search(query)
    context = "\n\n".join([doc.page_content for doc in results])
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = answer_chain.run({"question": query, "context": context})
    return response
    

def main():
    pdf_path = "./sample/war.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("DOCS:::::", docs)
    split_documents(docs)
    upload_to_qdrant(docs)
    
    while True:
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")
        
        question = input("Ask a question about the video (or type 'q' to quit):\n")
        if question.lower() == "q":
            break
        response = query_llm(question)
        print("🔹 Answering...")
        print("\nAnswer:\n", response)
   
if __name__ == "__main__":
    main()
 