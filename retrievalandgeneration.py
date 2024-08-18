from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.llms import Bedrock
import boto3
from langchain.prompts import PromptTemplate
from ingestion import get_vector_store,data_ingestion



bedrock = boto3.client(service_name="bedrock-runtime")


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)


def get_llm():
    llm = Bedrock(model_id = "amazon.titan-text-lite-v1",client=bedrock,model_kwargs={"max_tokens":512})


retriever = vectorstore_faiss.as_retriever(search_type="similarity",search_kwargs={"k":3})


def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(llm = llm,
                                     chain_type = "stuff",
                                     retriever = retriever,
                                     return_source_documents = True,
                                     chain_type_kwargs = {'prompt':PROMPT} )
    answer = qa({"query":query})
    return answer["result"]

if __name__ == '__main__':
    docs = data_ingestion()
    vectorstore_faiss = get_vector_store(docs)
    query = "What is RAG"
    llm = get_llm()
    get_response_llm(llm,vectorstore_faiss,query)