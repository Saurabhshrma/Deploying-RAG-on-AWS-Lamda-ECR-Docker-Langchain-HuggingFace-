import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbedding
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchian.vectorstores import FAISS
