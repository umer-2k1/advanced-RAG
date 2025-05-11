import csv
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from uuid import uuid4
import os


load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


COLLECTION_NAME = f"self-query-collection-{uuid4()}"
ollama_model = "llama3.2:latest"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(
    model=ollama_model,
)


qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vectordb = None


# Load the CSV file
df = pd.read_csv("product_catalog_data.csv")


documents = []
for _, row in df.iterrows():
    # Extract the page content
    page_content = row["page_content"]

    # Remove page_content from metadata
    metadata = row.drop("page_content").to_dict()

    # Convert features string to list if needed
    if "features" in metadata and isinstance(metadata["features"], str):
        metadata["features"] = metadata["features"].split(",")

    # Convert string boolean to actual boolean
    if "in_stock" in metadata and isinstance(metadata["in_stock"], str):
        metadata["in_stock"] = metadata["in_stock"].lower() == "true"

    # Convert numeric fields to appropriate types
    for field in [
        "price",
        "discount_percent",
        "screen_size",
        "stock_count",
        "rating",
        "reviews_count",
    ]:
        if field in metadata and pd.notna(metadata[field]):
            metadata[field] = float(metadata[field])

    documents.append(Document(page_content=page_content, metadata=metadata))

print(documents)
# model = OllamaLLM(model=ollama_model, streaming=True)
