from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

import os

load_dotenv()

""" loader = TextLoader("./index.md")
loader.load()
"""

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

file_path = os.path.join(os.path.dirname(__file__), "test.txt")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

def load_document(path: str):
    loader = TextLoader(path)
    return loader.load()

def split_document(document, chunk_size=1000, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    segments = splitter.split_documents(document)
    return segments

document = load_document(file_path)
docs = split_document(document)
db = Chroma.from_documents(docs, embeddings)
persist_directory = "chroma_db"
vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)
vectordb.persist()

query = "Explique como a fe cristã é uma fé racional"
matches = db.similarity_search(query, k=20)
print(matches[0])
answer =  chain.run(input_documents=matches, question=query)
print(answer)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
retrieval_chain.run(query)