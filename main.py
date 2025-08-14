from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('sample_data/one.pdf', mode='page')

docs = loader.load()
