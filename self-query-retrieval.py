import csv
import pandas as pd
from langchain.schema import Document

# from langchain_community.vectorstores import Qdrant
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.query_constructor.base import Comparator
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
import json
from langsmith import Client
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


COLLECTION_NAME = f"self-query-collection"
ollama_model = "llama3.2:latest"


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

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "rating": 9.9,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
        },
    ),
]

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OllamaEmbeddings(
    model=ollama_model,
)


if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# Initialize vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
# Store documents
# vector_store.add_documents(documents)
vector_store.add_documents(docs)


# Define metadata fields
# metadata_field_info = [
#     AttributeInfo(
#         name="price",
#         description="The price of the product in USD",
#         type="float",
#     ),
#     AttributeInfo(
#         name="category",
#         description="The product category",
#         type="string",
#     ),
#     AttributeInfo(name="brand", description="The brand of the product", type="string"),
#     AttributeInfo(
#         name="in_stock",
#         description="Whether the item is in stock",
#         type="boolean",
#     ),
#     AttributeInfo(
#         name="rating",
#         description="Customer rating from 1-5",
#         type="float",
#     ),
#     AttributeInfo(
#         name="release_date",
#         description="When the product was released, in ISO format",
#         type="timestamp",
#     ),
#     AttributeInfo(
#         name="features",
#         description="List of product features like Bluetooth, ANC, etc.",
#         type="list[string]",
#     ),
# ]

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

llm = OllamaLLM(model=ollama_model, streaming=True)


# SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    # document_contents="Product catalog entries",
    document_contents="Brief summary of a movie",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True,
    # allowed_comparators=[
    #     Comparator.EQ,
    #     Comparator.NE,
    #     Comparator.GT,
    #     Comparator.GTE,
    #     Comparator.LT,
    #     Comparator.LTE,
    #     Comparator.CONTAIN,
    #     Comparator.LIKE,
    #     Comparator.IN,
    #     Comparator.NIN,
    # ],
)


while True:
    print("\n\n ----------------------------------")
    print("\n\n ----------------------------------")
    print("\n\n ----------------------------------")

    # question = input("Ask a question about the video (or type 'q' to quit):\n")

    # print("\n\n")
    # if question.lower() == "q":
    #     break

    result = retriever.invoke("I want to watch a movie rated higher than 8.5")
    print
    for doc in result:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")
