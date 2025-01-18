import google.generativeai as genai
from RAG import index_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import numpy as np
import os
import pickle


def embed_input(pdf_path):
    loader = PyPDFLoader(pdf_path)
    raw_text = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_text)

    embeddings = []
    for chunk in chunks:
        embeddings.append(chunk.page_content)
    embeddings = np.array(embeddings)
    return embeddings

def get_or_create_vectorstore(vectorstore_path, ref_labels):
    if os.path.exists(vectorstore_path):
        with open(vectorstore_path, 'rb') as f:
            return pickle.load(f)
    print('Generating vectorstore')
    vectorstore = index_documents(ref_labels)
    with open(vectorstore_path, 'wb') as f:
        pickle.dump(vectorstore, f)
        print('Vectorstore saved for future reference')
    return vectorstore

def retrieve_similar_docs(doc_path):
    if not os.path.exists(doc_path):
        raise ValueError('Incorrect file path')
    
    doc_data = embed_input(doc_path)
    vectorstore_path = 'ReferenceData.pkl'
    ref_labels = pd.read_csv('reference_data.csv')
    vectorstore = get_or_create_vectorstore(vectorstore_path, ref_labels)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    similar_docs = retriever.invoke(doc_data)

    return similar_docs


def conclusion(doc_path):

    similar_docs = retrieve_similar_docs(doc_path)

    genai.configure(api_key="API_KEY_HERE")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                system_instruction="""Being a very good conference chair, you have to tell if the given research paper is publishable. 
                                Respond in format : 'Yes/No. Conference Names(s).Reasoning: Lorem epsum ...'""")

    prompt = f"""If it is publishable, which conference(s) would be most relevant among CVPR, EMNLP, KDD, NeurIPS and TMLR.
    Consider all aspects of the given paper to reach the final decision. Prefer declining the conference if unsure. 
    Give strong convincing reasoning. Reasoning must add some research paper **titles** as examples to confirm that each chosen conference is relevant,
    by comparing the domain, quality and quantity of content provided in them with that given in this paper.
    Use minimal words and a single paragraph. Useful references: {similar_docs}"""
    # print('Sending Prompt')

    with open(doc_path, 'rb') as file:
        pdf_data = file.read()
    response = model.generate_content([{'mime_type': 'application/pdf', 'data': pdf_data}, prompt])
    return(response.text)