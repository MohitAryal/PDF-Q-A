# RAG-based PDF Context Retriever

This project demonstrates a **Retrieval-Augmented Generation (RAG)**
pipeline that extracts context from PDF documents using **LangChain**,
**ChromaDB**, **HuggingFace embeddings**, and **Groq LLM**.\
It allows users to query the contents of PDFs with precise,
context-aware answers.

------------------------------------------------------------------------

##  Features

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

##  Project Structure

    project/
    │── sample_data/        # Store PDFs here (e.g., one.pdf)
    │── main.py             # RAG pipeline implementation
    │── requirements.txt    # Dependencies
    │── README.md           # Documentation

------------------------------------------------------------------------

##  Installation

``` bash
# Clone this repository
git clone https://github.com/MohitAryal/PDF-Q-A
cd PDF-Q-A

# Create and activate virtual environment
python -m venv venv
source venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Environment Setup

Create a `.env` file in the project root and add your Groq API key:

``` env
GROQ_API_KEY=your_groq_api_key
```

------------------------------------------------------------------------

## Usage

Place your PDF file inside the `sample_data` folder.\
For example: `sample_data/YourPDF.pdf`.

------------------------------------------------------------------------

## Example Output

    Query: Which vector DB is user-friendly?
    Answer: ChromaDB is considered one of the most user-friendly vector databases due to its simplicity and ease of integration.

------------------------------------------------------------------------

## Requirements

-   Python 3.10+
-   LangChain
-   LangChain Chroma
-   HuggingFace Transformers
-   LangChain Groq
-   dotenv

Install all dependencies via:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------
