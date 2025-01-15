# Summary-Based RAG System

A Retrieval-Augmented Generation (RAG) system that uses text summaries for enhanced semantic search capabilities. The system creates embeddings from document content combined with their summaries to improve search relevance.

## Features

- Vector similarity search using FAISS
- OpenAI embeddings for semantic search
- Summary generation using GPT-4
- Rich CLI interface with colored output
- Efficient batch processing of embeddings
- Persistent vector database storage

## Setup

1. Install dependencies:
```bash
pip install openai faiss-cpu rich
```

2. Set up your OpenAI API key:
- Open `main.py` and `get_summary.py`
- Add your OpenAI API key to the `API_KEY` variable

3. Prepare your data:
- Place your text data in `data/data.txt`
- Each document should be separated by double newlines
- First line of each document should be its title

## Project Structure

- `vectordb.py`: Vector database implementation using FAISS
- `utils.py`: Helper functions for CLI interface
- `splitter.py`: Data preprocessing script
- `get_summary.py`: Summary generation using OpenAI
- `main.py`: Main application interface

## How it Works

1. Documents are first split and structured from raw text
2. GPT-4 generates concise summaries for each document
3. Documents are embedded using OpenAI's text-embedding-3-small model
4. Embeddings are stored in a FAISS vector database
5. User queries are converted to embeddings and matched against the database
6. Results are displayed with similarity scores and formatted output