# packages
import os
import re
from pprint import pprint
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
import openai
from openai import OpenAI

def parse_add_to_collection(pdf, collection):
    
    print(f"Processing file {pdf}")
    reader = PdfReader(pdf)
    ipcc_texts = [page.extract_text().strip() for page in reader.pages]
    
    # remove all header / footer texts
    ipcc_wo_header_footer = [re.sub(r'\d+\nTechnical Summary', '', s) for s in ipcc_texts]
    # remove \nTS
    ipcc_wo_header_footer = [re.sub(r'\nTS', '', s) for s in ipcc_wo_header_footer]

    # remove TS\n
    ipcc_wo_header_footer = [re.sub(r'TS\n', '', s) for s in ipcc_wo_header_footer]

    char_splitter = RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0.2
        )

    texts_char_splitted = char_splitter.split_text('\n\n'.join(ipcc_wo_header_footer))
    
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0.2,
        tokens_per_chunk=256
        )

    texts_token_splitted = []
    for text in texts_char_splitted:
        try:
            texts_token_splitted.extend(token_splitter.split_text(text))
        except:
            print(f"Error in text: {text}")
            continue

    # Add Documents
    ids = [str(i) for i in range(len(texts_token_splitted))]
    collection.add(
        ids=ids,
        documents=texts_token_splitted
        )
 
import os

chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("ipcc")

# Define the directory containing the PDF files
pdf_dir = "solvecube-corpus"

# Iterate over each PDF file in the directory
# Iterate over each PDF file in the directory
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        try:
            parse_add_to_collection(pdf_path, chroma_collection)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
