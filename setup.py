from setuptools import find_packages, setup

setup(
    name="qasystem",
    version="0.0.1",
    author="Saurabh",
    author_email="saurabh1346.ss@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain==0.2.14",
        "langchainhub==0.1.21",  
        "beautifulsoup4==4.12.3",
        "tiktoken==0.7.0",
        "boto3==1.34.37",
        "langchain-community==0.2.12",
        "awscli==2.17.32",
        "streamlit==1.37.0",
        "pypdf==3.0.1",
        "faiss-cpu==1.8.0.post1",
        "langchain-aws==0.1.16"
    ]
)