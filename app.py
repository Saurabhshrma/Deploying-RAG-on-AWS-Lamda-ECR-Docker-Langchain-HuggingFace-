import json
import os 
import sys
import boto3
import streamlit as st


from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS

from ingestion import data_ingestion,get_vector_store

from retrievalandgeneration import get_llm,get_response_llm

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

# def main():
#     st.set_page_config("QA with Saurabh's Resume")
#     st.header("QA with Doc using langchain and AWSBedrock")
    
#     user_question=st.text_input("Ask a question")
    
#     with st.sidebar:
#         st.title("Create the vector store")
#         if st.button("vectors update"):
#             with st.spinner("processing..."):
#                 docs=data_ingestion()
#                 get_vector_store(docs)
#                 st.success("done")
                
#         if st.button("Titan model"):
#             with st.spinner("processing..."):
#                 faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
#                 llm=get_llm()
#                 st.write(get_response_llm(llm,faiss_index,user_question)) 
#                 st.success("Done")
    
# if __name__=="__main__":
#     #this is my main method
#     main()

def main():
    st.set_page_config("QA with Saurabh's Resume")

    # Main Content Area
    st.title("Ask Questions About Saurabh's Resume")
    st.write("Leveraging LangChain and AWS Bedrock, this app provides answers based on Saurabh's resume.")

    user_question = st.text_input("Enter your question here:")

    # Sidebar
    st.sidebar.title("Vector Store Management")
    st.sidebar.write("Control the vector store used for answering questions.")

    if st.sidebar.button("Update Vectors"):
        with st.spinner("Processing..."):
            docs = data_ingestion()
            get_vector_store(docs)
        st.sidebar.success("Vector store updated!")

    if st.sidebar.button("Submit") and user_question:  # Check if a question is entered
        with st.spinner("Processing...Running Titan..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm()
            response = get_response_llm(llm, faiss_index, user_question)

        # Display the answer below the question
        st.write("**Question:**", user_question)
        st.write("**Answer:**", response)
        st.success("Done!")

if __name__ == "__main__":
    main()    