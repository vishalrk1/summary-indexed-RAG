import os
import pickle
import json
import faiss 
import numpy as np
from openai import OpenAI
from rich.console import Console
from sklearn.preprocessing import normalize

class IVFPQVectorDB:
    def __init__(self, api_key=None, console=None, d=1536):
        """Initialize the vector database"""
        self.client = OpenAI(api_key=api_key)
        self.index = None
        self.metadata = []
        self.query_cache = {}
        self.db_path = "data/advance_vector_db.pkl"
        self.console = console if console else Console()
        self.dimension = d
        
        # PCA and index parameters
        self.target_reduction = d // 24
        self.pca_dimension = self.target_reduction - (self.target_reduction % 4)  # Ensure divisible by 4
        self.pca_matrix = None
        self.M = None  # Number of sub-quantizers
        self.bits_per_subvector = None
        self.nlist = None  # Number of clusters
        self.raw_index = None  # Store original vectors

    def _calculate_nlist(self, n_vectors):
        """Calculate appropriate number of clusters based on dataset size"""
        if n_vectors < 1000:
            nlist = max(1, int(np.sqrt(n_vectors)))
        else:
            nlist = min(4096, max(1, n_vectors // 50))
        return max(1, min(nlist, n_vectors // 4))

    def _recalculate_pca_params(self, n_vectors):
        """Recalculate PCA and index parameters based on data size"""
        # Adjust PCA dimension if needed
        if n_vectors < self.target_reduction:
            self.pca_dimension = max(4, (n_vectors // 4) * 4)
            self.console.print(
                f"Adjusted PCA dimension to {self.pca_dimension} for {n_vectors} vectors",
                style="bold yellow"
            )
        possible_m = [v for v in [4, 8, 16] if self.pca_dimension % v == 0]
        if not possible_m:
            old_dim = self.pca_dimension
            self.pca_dimension = ((self.pca_dimension // 4) * 4)
            self.console.print(f"Adjusted PCA dimension from {old_dim} to {self.pca_dimension}")
            possible_m = [v for v in [4, 8, 16] if self.pca_dimension % v == 0]
            
        self.M = possible_m[0]
        self.bits_per_subvector = min(8, max(4, self.pca_dimension // (self.M * 4)))
        self.nlist = self._calculate_nlist(n_vectors)

    def _create_index(self, n_vectors):
        """Create FAISS index with current parameters"""
        if any(v is None for v in [self.M, self.bits_per_subvector, self.nlist]):
            self._recalculate_pca_params(n_vectors)

        self.pca_matrix = faiss.PCAMatrix(self.dimension, self.pca_dimension, 0, False)
        quantizer = faiss.IndexFlatIP(self.pca_dimension)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.pca_dimension,
            self.nlist,
            self.M,
            self.bits_per_subvector
        )
        self.index.make_direct_map()
        self.raw_index = faiss.IndexFlatIP(self.dimension)
        
        self.console.print(
            f"Created index with: PCA={self.pca_dimension}, nlist={self.nlist}, "
            f"M={self.M}, bits={self.bits_per_subvector}",
            style="bold blue"
        )

    def _get_embedding(self, text):
        """Get embedding for a single text"""
        res = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return res.data[0].embedding

    def _get_batch_embeddings(self, texts, batch_size=128):
        """Get embeddings for a batch of texts"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = self.client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [item.embedding for item in res.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _process_vectors(self, vectors, batch_size=10000):
        """Process vectors through PCA reduction"""
        if vectors.shape[0] == 0:
            raise ValueError("Empty vector array provided")
            
        vectors = normalize(vectors, axis=1, norm='l2')
        
        if self.pca_matrix is None:
            self._create_index(vectors.shape[0])
        
        if not self.pca_matrix.is_trained:
            n_vectors = vectors.shape[0]
            training_vectors = vectors[:min(n_vectors, batch_size)]
            
            try:
                self.pca_matrix.train(training_vectors)
            except RuntimeError as e:
                self.console.print(f"PCA training failed: {e}", style="bold red")
                self._recalculate_pca_params(n_vectors)
                self._create_index(n_vectors)
                self.pca_matrix.train(training_vectors)
        
        reduced_vectors = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            reduced_batch = self.pca_matrix.apply_py(batch)
            reduced_vectors.append(reduced_batch)
        
        return np.vstack(reduced_vectors)

    def load_data(self, data_file):
        """Load and process data into the index"""
        with self.console.status("Loading data...") as status:
            if self.index is not None and len(self.metadata) > 0:
                self.console.print("Data already loaded.")
                return
            
            if os.path.exists(self.db_path):
                self.load_db()
                return
            
            # Load data
            with open(data_file, 'r') as file:
                data = json.load(file)
            
            texts = [f"Title: {item['title'].lower()}\n\nInformation: {item['content'].lower()}\n\nSummary: {item['summary'].lower()}" for item in data]
                    
            batch_size = 128
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_batch_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(all_embeddings).astype('float32')
            
            status.update("Processing vectors...")
            processed_embeddings = self._process_vectors(embeddings_array)
            
            status.update("Training index...")
            if not self.index.is_trained:
                self.index.train(processed_embeddings)
            
            status.update("Adding vectors to index...")
            self.index.add(processed_embeddings)
            self.raw_index.add(embeddings_array)
            
            self.metadata = data
            self.save_db()
        
        self.console.print("Data loaded successfully.", style="bold green")

    def save_db(self):
        """Save the database to disk"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        faiss_path = self.db_path.replace('.pkl', '.faiss')
        raw_faiss_path = self.db_path.replace('.pkl', '_raw.faiss')
        pca_path = self.db_path.replace('.pkl', '_pca.faiss')
        
        faiss.write_index(self.index, faiss_path)
        faiss.write_index(self.raw_index, raw_faiss_path)
        faiss.write_VectorTransform(self.pca_matrix, pca_path)
        
        data = {
            "metadata": self.metadata,
            "query_cache": self.query_cache
        }
        
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        """Load the database from disk"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Database file not found")
            
        faiss_path = self.db_path.replace('.pkl', '.faiss')
        raw_faiss_path = self.db_path.replace('.pkl', '_raw.faiss')
        pca_path = self.db_path.replace('.pkl', '_pca.faiss')
        
        if not (os.path.exists(faiss_path) and os.path.exists(raw_faiss_path) and os.path.exists(pca_path)):
            raise FileNotFoundError("One or more FAISS files are missing")

        self.index = faiss.read_index(faiss_path)
        self.raw_index = faiss.read_index(raw_faiss_path)
        self.pca_matrix = faiss.read_VectorTransform(pca_path)
        
        with open(self.db_path, 'rb') as file:
            data = pickle.load(file)
            self.metadata = data.get('metadata', [])
            self.query_cache = data.get('query_cache', {})
            
    def search(self, query, k=3, similarity_threshold=0.5, nprobe=32, pre_k=10):
        with self.console.status("Searching...") as cn:
            if self.index is None:
                self.console.print("Index not loaded.", style="bold red")
                return
            
            if query in self.query_cache:
                query_embedding = self.query_cache[query]
            else:
                query_embedding = self._get_embedding(query)
                self.query_cache[query] = query_embedding
            
            query_embedding = np.array([query_embedding]).astype('float32')
            query_embedding = normalize(query_embedding, axis=1, norm="l2")
            processed_query = self.pca_matrix.apply(query_embedding)
            
            # setting search parameters
            if isinstance(self.index, faiss.IndexIVFPQ):
                self.index.nprobe = nprobe
            
            # first search
            similarity, indices = self.index.search(processed_query, pre_k)
            
            # reranking using exact distance
            if indices.size > 0:
                candidates = indices[0]
                valid_candidates = candidates[candidates != -1]

                if len(valid_candidates) > 0:
                    candidate_vectors = np.empty((len(valid_candidates), self.dimension), dtype=np.float32)
                    for i, c in enumerate(valid_candidates):
                        candidate_vectors[i] = self.raw_index.reconstruct(int(c))
                    
                    exact_similarities = np.dot(query_embedding, candidate_vectors.T)[0]
                    
                    # sorting 
                    reranked_indices = valid_candidates[np.argsort(-exact_similarities)]
                    reranked_similarities = np.sort(exact_similarities)[::-1]
                    
                    # filtering based on threshold
                    top_examples = []
                    for similarity_score, index in zip(reranked_similarities, reranked_indices):
                        if similarity_score >= similarity_threshold and index < len(self.metadata):
                            example = {
                                "metadata": self.metadata[int(index)],
                                "similarity": float(similarity_score)
                            }
                            top_examples.append(example)
                            if len(top_examples) >= k:
                                break
                    return top_examples
            return []
