# RAG-based PDF Context Retriever

This project demonstrates a **Retrieval-Augmented Generation (RAG)**
pipeline that extracts context from PDF documents using **LangChain**,
**ChromaDB**, **HuggingFace embeddings**, and **Groq LLM**.\
It allows users to query the contents of PDFs with precise,
context-aware answers.

------------------------------------------------------------------------

## 🚀 Features

-   Loads PDF documents using `PyPDFLoader`.
-   Splits documents into manageable chunks with
    `RecursiveCharacterTextSplitter`.
-   Embeds text using `sentence-transformers/all-MiniLM-L6-v2` from
    HuggingFace.
-   Stores vector embeddings in **ChromaDB** for fast retrieval.
-   Uses **Groq LLM** (`deepseek-r1-distill-llama-70b`) for contextual
    Q&A.
-   Answers are restricted strictly to the PDF context.

------------------------------------------------------------------------

## 📂 Project Structure

    project/
    │── sample_data/        # Store PDFs here (e.g., one.pdf)
    │── chroma/             # Persistent ChromaDB storage
    │── main.py             # RAG pipeline implementation
    │── requirements.txt    # Dependencies
    │── README.md           # Documentation

------------------------------------------------------------------------

## 📦 Installation

``` bash
# Clone this repository
git clone <your-repo-link>
cd <your-repo-name>

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ⚙️ Environment Setup

Create a `.env` file in the project root and add your Groq API key:

``` env
GROQ_API_KEY=your_groq_api_key
```

------------------------------------------------------------------------

## 🛠️ Usage

Place your PDF file inside the `sample_data` folder.\
For example: `sample_data/one.pdf`.

``` python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load PDF
loader = PyPDFLoader("sample_data/one.pdf", mode="single")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in ChromaDB
vector_store = Chroma(collection_name="Pdf", embedding_function=embeddings, persist_directory="chroma")
vector_store.add_documents(chunks)

# Create retriever
retriever = VectorStoreRetriever(vectorstore=vector_store)

# Define prompt
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant responding to user queries using the following context. {context}"),
    ("human", "{input}")
])

# Initialize model
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, reasoning_format="hidden")

# Create retrieval chain
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

# Query
response = retrieval_chain.invoke({"input": "Which vector DB is user-friendly?"})
print(response["answer"])
```

------------------------------------------------------------------------

## 📊 Example Output

    ChromaDB is considered one of the most user-friendly vector databases due to its simplicity and ease of integration.

------------------------------------------------------------------------

## 📌 Requirements

-   Python 3.10+
-   LangChain
-   ChromaDB
-   HuggingFace Transformers
-   Groq API access
-   dotenv

Install all dependencies via:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License.
