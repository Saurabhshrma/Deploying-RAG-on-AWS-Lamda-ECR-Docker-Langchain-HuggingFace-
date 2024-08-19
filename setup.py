from setuptools import find_packages, setup

setup(
    name="qasystem",
    version="0.0.1",
    author="Saurabh",
    author_email="saurabh1346.ss@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchainhub",  
        "beautifulsoup4",
        "tiktoken",
        "boto3",
        "langchain-community",
        "awscli",
        "streamlit",
        "pypdf",
        "faiss-cpu",
        "langchain-aws"
    ]
)