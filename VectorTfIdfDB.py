import os
import pickle
import json
import faiss
import numpy as np
from openai import OpenAI
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse


class VectorTIdfDB:
    def __init__(self, console=None, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.index = None
        self.metadata = []
        self.query_cache = {}
        self.db_path = "data/vector_tfidf_db.pkl"
        self.console = console if console else Console()

        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            token_pattern=r'\b\w+\b',
            max_features=10000
        )
        self.tfidf_matrix = None
        self.documents = []

    def _preprocess_text(self, item):
        """Combine title, content and summary for TF-IDF processing"""
        return f"{item['title']} {item['content']} {item['summary']}".lower()
      
    def _create_index(self):
        res = self.client.embeddings.create(
            input="hello",
            model="text-embedding-3-small"
        )
        print("Embedding size:", len(res.data[0].embedding))
        self.index = faiss.IndexFlatIP(len(res.data[0].embedding))
    
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
    
    def _get_embedding(self, text):
        res = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return res.data[0].embedding
    
    def load_db(self):
        if not os.path.exists(self.db_path):
            print("Database file not found.")
            return
    
        faiss_path = self.db_path.replace(".pkl", ".faiss")
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        
        if os.path.exists(self.tfidf_matrix_path):
            self.tfidf_matrix = sparse.load_npz(self.tfidf_matrix_path)
        
        with open(self.db_path, 'rb') as file:
            data = pickle.load(file)

        self.metadata = data['metadata']
        self.query_cache = data['query_cache']
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.documents = data['documents']

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        faiss_path = self.db_path.replace('.pkl', '.faiss')
        faiss.write_index(self.index, faiss_path)

        if self.tfidf_matrix is not None:
            sparse.save_npz(self.db_path.replace('.pkl', '.npz'), self.tfidf_matrix)

        data = {
            "metadata": self.metadata,
            "query_cache": self.query_cache,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "documents": self.documents
        }

        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_data(self, data_file):
        with self.console.status("Loading data from data.json"):
            if self.index is not None and len(self.metadata) > 0:
                self.console.print("Data already loaded. Skipping...")
                return
            
            if os.path.exists(self.db_path):
                self.load_db()
                return

            if self.index is None:
                self._create_index()
            
            with open(data_file, 'r') as file:
                data = json.load(file)

            self.documents = [self._preprocess_text(item) for item in data]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
            
            texts = [f"Title: {item['title'].lower()}\n\nInformation: {item['content'].lower()}\n\nSummary: {item['summary'].lower()}" for item in data]
            embeddings = self._get_batch_embeddings(texts)

            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            self.metadata = data

            self.save_db()
            self.console.print("Data loaded successfully.")
    
    def search(self, query, k=3, similarity_threshold=0.5, vector_weight=0.7):
        with self.console.status("Searching for similar documents...."):
            if self.index is None or self.tfidf_matrix is None:
                self.console.print("Data not loaded. Please load data first.")
                return
            
            query_embedding = self.query_cache.get(query)
            if query_embedding is None:
                query_embedding = self._get_embedding(query)
                self.query_cache[query] = query_embedding
            
            query_embedding_array = np.array([query_embedding]).astype('float32')
            similarity, indices = self.index.search(query_embedding_array, len(self.metadata))
            vector_similarities = similarity[0]

            tfidf_query = self.tfidf_vectorizer.transform([query])
            tfidf_similarities = cosine_similarity(tfidf_query, self.tfidf_matrix).flatten()

            if len(vector_similarities) > 0:
                vector_similarities = (vector_similarities - vector_similarities.min()) / (vector_similarities.max() - vector_similarities.min() + 1e-10)
            if len(tfidf_similarities) > 0:
                tfidf_similarities = (tfidf_similarities - tfidf_similarities.min()) / (tfidf_similarities.max() - tfidf_similarities.min() + 1e-10)

            combined_scores = (vector_weight * vector_similarities + 
                             (1 - vector_weight) * tfidf_similarities)
            top_indices = np.argsort(combined_scores)[::-1]
            
            results = []
            for idx in top_indices:
                if combined_scores[idx] >= similarity_threshold:
                    result = {
                        "metadata": self.metadata[idx],
                        "similarity": float(combined_scores[idx]),
                        "vector_similarity": float(vector_similarities[idx]),
                        "tfidf_similarity": float(tfidf_similarities[idx])
                    }
                    results.append(result)
                    if len(results) >= k:
                        break

            return results
        