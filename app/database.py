# import
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import os
import uuid


""" 
1. Initialize the database with a embedding function and disk location
2. Create function to add document to database
3. Create function to query database
"""

class Database:
    def __init__(self, embedding_function, disk_location):
        self.embedding_function = embedding_function
        self.db = Chroma(embedding_function=embedding_function, persist_directory=disk_location)

    def _generate_id(self):
        return str(uuid.uuid4())

    def add_documents(self, docs, ids):    
        id = self.db.add_documents(docs, ids=ids)
        return id
    
    def similarity_search(self, query, k=10):
        docs = self.db.similarity_search(query, k=k)
        return docs
    
    def _count_documents(self):
        return self.db._collection.count()
    
class DocumentManager:
    def __init__(self, database: Database, text_splitter):
        self.text_splitter = text_splitter
        self.db = database

    def add_document(self, file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)
        generated_ids = [self.db._generate_id() for _ in docs]
        result = self.db.add_documents(docs, ids=generated_ids)
        return result
    
    def similarity_search(self, query):
        docs = self.db.similarity_search(query)
        return docs

if __name__ == "__main__":
    # initialize database
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Database(embedding_function=embedding_function, disk_location="./chroma_db")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    document_manager = DocumentManager(database=db, text_splitter=text_splitter)

    print("Count documents in database", db._count_documents())

    cwd = os.getcwd()
    file_path = os.path.join(cwd, "transcriptions", "transcription0.txt")

    # add document to database
    result = document_manager.add_document(file_path)
    print("Document added to database", result)

    # query database
    query = "O primeiro passo de iniciação na maioria das religiões de matriz afro consiste na descoberta do ancestral. A partir da introdução do livro de Fatumbi; qual o objetivo dessa associação?"

    docs = document_manager.similarity_search(query)
    print("Documents retrieved from database", docs)