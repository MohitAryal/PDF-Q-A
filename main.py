from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
import re
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('sample_data/one.pdf', mode='single')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma(collection_name='Pdf', embedding_function=embeddings, persist_directory='chroma')
vector_store.add_documents(chunks)

retriever = VectorStoreRetriever(vectorstore=vector_store)

prompt = ChatPromptTemplate([
    ('system', 'You are a helpful assistant responding to user queries using the following context. {context}'),
    ('human', '{input}')
])
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

response = retrieval_chain.invoke({'input': 'Which vector DB is user-friendly?'})
clean_answer = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
print(clean_answer)