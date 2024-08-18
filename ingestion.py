from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbedding
from langchain.llms.bedrock import Bedrock
from langchian.vectorstores import FAISS
import boto3

import json
import os
import sys

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbedding(model_id="amazon.titan-embed-text-v1",client=bedrock)


def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    return docs


def get_vector_store(docs):
    vector_store_faiss = FAISS.from_document(docs,bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")

if __name__ == 'main':
    docs = data_ingestion()
    get_vector_store(docs)