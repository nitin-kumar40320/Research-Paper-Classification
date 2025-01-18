import os
import pandas as pd
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import requests
import numpy as np
from tenacity import retry, stop_after_delay, wait_exponential

def load_and_preprocess(dataframe):
    documents = []
    for _, row in dataframe.iterrows():
        conference_value = row.get('conference')
        if pd.isna(conference_value):
            # Exclude 'conference' from the path if it's NaN
            pdf_path = os.path.join('Reference', str(row['label']), str(row['path']))
        else:
            # Include 'conference' in the path
            pdf_path = os.path.join('Reference', str(row['label']), str(conference_value), str(row['path']))

        loader = PyPDFLoader(pdf_path)
        raw_text = loader.load()
        
        # Split text for better indexing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(raw_text)
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata.update({
                'title': row['heading'],
                'label': row['label'],
                'conference': row.get('conference', None)
            })
            documents.append(chunk)
    return documents


def index_documents(dataframe):
    documents = load_and_preprocess(dataframe)
    if not documents:
        raise ValueError("No documents to process. Please check the input dataframe and preprocessing steps.")

    # Create a FAISS vectorstore
    embedding_class = Embedding_generator()
    vectorstore = FAISS.from_documents(documents, embedding_class)

    return vectorstore

API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
headers = {"Authorization": "Bearer API_KEY_HERE"}
class Embedding_generator(Embeddings):

    @retry(stop=stop_after_delay(300), wait=wait_exponential(multiplier=1, min=60, max=120))
    def embed_documents(self, texts):
        response = requests.post(API_URL, headers=headers, json={"inputs": texts})
        # Convert to NumPy array for FAISS
        embeddings_array = np.array(response.json(), dtype=np.float32)
        embeddings_array = embeddings_array.reshape(-1, embeddings_array.shape[-1])
        return embeddings_array
    
    @retry(stop=stop_after_delay(300), wait=wait_exponential(multiplier=1, min=60, max=120))
    def embed_query(self, texts):
        response = requests.post(API_URL, headers=headers, json={"inputs": str(texts)})
        # Convert to NumPy array for FAISS
        embeddings_array = np.array(response.json(), dtype=np.float32)
        return embeddings_array
