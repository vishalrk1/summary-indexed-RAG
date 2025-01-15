import os
import pickle
import json
import faiss
import numpy as np
from openai import OpenAI
from rich.console import Console

class VectorDB:
    def __init__(self, console=None, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.index = None
        self.metadata = []
        self.query_cache = {}
        self.db_path = "data/vector_db.pkl"
        self.console = console if console else Console()

    def _create_index(self):
        res = self.client.embeddings.create(
            input="hello",
            model="text-embedding-3-small"
        )
        print("Embedding size:", len(res.data[0].embedding))
        self.index = faiss.IndexFlatIP(len(res.data[0].embedding))
    
    def _get_embedding(self, text):
        res = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return res.data[0].embedding
    
    def _get_batch_embeddings(self, texts, batch_size=128):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[:i+batch_size]
            res = self.client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [item.embedding for item in res.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings
    
    def load_db(self):
        if not os.path.exists(self.db_path):
            print("Database file not found.")
            return
        
        # finding faiss db path 
        faiss_path = self.db_path.replace(".pkl", ".faiss")
        self.index = faiss.read_index(faiss_path)

        with open(self.db_path, 'rb') as file:
            data = pickle.load(file)

        self.metadata = data['metadata']
        self.query_cache = data['query_cache']

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        faiss_path = self.db_path.replace('.pkl', '.faiss')
        faiss.write_index(self.index, faiss_path)
        
        data = {
            "metadata": self.metadata,
            "query_cache": self.query_cache,
        }
        
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_data(self, data_file):
        with self.console.status("Loading Data..."):
            if self.index is not None and len(self.metadata) > 0:
                print("Data already loaded.")
                return
            
            if os.path.exists(self.db_path):
                self.load_db()

            if self.index is None:
                self._create_index()

            with open(data_file, 'r') as file:
                data = json.load(file)

            texts = [f"Title: {item['title'].lower()}\n\nInformation: {item['content'].lower()}\n\nSummary: {item['summary'].lower()}" for item in data]
            embeddings = self._get_batch_embeddings(texts)

            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            self.metadata = data

            self.save_db()
            print("Vector Data loaded successfully.")

    def search(self, query, k=3, similarity_threshold=0.5):
        with self.console.status("Searching..."):
            if self.index is None:
                raise ValueError("Index not found. Load the data first.")

            # qetting query embeddings
            if query in self.query_cache:
                query_embedding = self.query_cache[query]
            else:
                query_embedding = self._get_embedding(query)
                self.query_cache[query] = query_embedding
            
            query_embedding_array = np.array([query_embedding]).astype('float32')

            # Similarity search
            top_examples = []
            similarity, indices = self.index.search(query_embedding_array, len(self.metadata))
            for similarity_score, index in zip(similarity[0], indices[0]):
                if similarity_score >= similarity_threshold and index != -1 and index < len(self.metadata):
                    example = {
                        "metadata": self.metadata[index],
                        "similarity": float(similarity_score)
                    }
                    top_examples.append(example)
                    if len(top_examples) >= k:
                        break  
            
        return top_examples[:k]
