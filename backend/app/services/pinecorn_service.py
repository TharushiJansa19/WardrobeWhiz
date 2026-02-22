from pinecone import Pinecone, PodSpec
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
)
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
import os
from dotenv import load_dotenv
from typing import List
from flask import current_app as app


def get_service_context():
    # Use the api_key to initialize your model here
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=app.config['GOOGLE_API_KEY']
    )

    service_context = ServiceContext.from_defaults(
        llm=Gemini(api_key=app.config['GOOGLE_API_KEY']),
        embed_model=embed_model,
    )

    return service_context


def get_storage_context():
    # Use the api_key to initialize your model here
    pinecone = Pinecone(api_key=app.config['PINECONE_API_KEY'])

    # Check if the index exists, if not, create one
    index_exists = any(index['name'] == app.config['PINECONE_INDEX_NAME'] for index in pinecone.list_indexes())
    if not index_exists:
        print(app.config['PINECONE_INDEX_NAME'], 'PINECONE_INDEX_NAME', pinecone.list_indexes())
        # pinecone.create_index(name=app.config['PINECONE_INDEX_NAME'], dimension=768)
        pinecone.create_index(
            name=app.config['PINECONE_INDEX_NAME'],
            dimension=768,
            metric="cosine",
            spec=PodSpec(
                environment="gcp-starter"
            )
        )

    # Connect to your Pinecone index
    index = pinecone.Index(app.config['PINECONE_INDEX_NAME'])

    storage_context = StorageContext.from_defaults(
        vector_store=PineconeVectorStore(index)
    )

    return storage_context


def get_index():
    pinecone = Pinecone(api_key=app.config['PINECONE_API_KEY'])

    # Check if the index exists, if not, create one
    index_exists = any(index['name'] == app.config['PINECONE_INDEX_NAME'] for index in pinecone.list_indexes())
    if not index_exists:
        print(app.config['PINECONE_INDEX_NAME'], 'PINECONE_INDEX_NAME', pinecone.list_indexes())
        # pinecone.create_index(name=app.config['PINECONE_INDEX_NAME'], dimension=768)
        pinecone.create_index(
            name=app.config['PINECONE_INDEX_NAME'],
            dimension=768,
            metric="cosine",
            spec=PodSpec(
                environment="gcp-starter"
            )
        )

    # Connect to your Pinecone index
    index = pinecone.Index(app.config['PINECONE_INDEX_NAME'])

    return index


def insert_into_pinecone(nodes: List[dict]):
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=get_storage_context(),
        service_context=get_service_context(),
    )


def get_similar_records(nodes: List[dict]):
    index = get_index()

    summary = nodes[0].text

    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=app.config['GOOGLE_API_KEY']
    )

    embeddings = embed_model.get_text_embedding(summary)
    top_k = 10

    results = index.query(
        vector=embeddings,
        top_k=top_k,
        include_metadata=True
    )

    return results


def get_similar_records_by_text(text: str):
    index = get_index()

    summary = text

    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=app.config['GOOGLE_API_KEY']
    )

    embeddings = embed_model.get_text_embedding(summary)
    top_k = 10

    results = index.query(
        vector=embeddings,
        top_k=top_k,
        include_metadata=True
    )

    return results