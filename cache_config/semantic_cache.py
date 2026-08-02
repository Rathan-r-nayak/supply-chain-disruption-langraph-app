import os
import pickle
import time
import numpy as np
from Utils.logger import get_logger

logger = get_logger("LOCAL_SEMANTIC_CACHE")

class LocalSemanticCache:
    def __init__(self, embedding_model, ttl_seconds: int = 3600, similarity_threshold: float = 0.92, cache_file: str = "cache_config/semantic_cache.pkl"):
        self.embedder = embedding_model
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.cache = []

        # 🌟 Ensure folder exists and load disk cache on startup
        self._ensure_dir_exists()
        self._load_from_disk()

    def _ensure_dir_exists(self):
        """Creates the parent folder (e.g., 'cache_config/') if it doesn't exist."""
        dirname = os.path.dirname(self.cache_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def _save_to_disk(self):
        """Persists the in-memory cache list to disk."""
        try:
            self._ensure_dir_exists()
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save semantic cache to disk: {e}")

    def _load_from_disk(self):
        """Loads cached vectors from disk on startup."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"📂 Loaded {len(self.cache)} items from cache file '{self.cache_file}'")
            except Exception as e:
                logger.error(f"Failed to load semantic cache file: {e}")
                self.cache = []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def get(self, query: str, current_user_id: str) -> str | None:
        """
        Retrieves a response ONLY if:
        1. TTL is valid
        2. Cosine similarity >= threshold
        3. Entry is GLOBAL OR belongs to current_user_id
        """
        now = time.time()
        original_len = len(self.cache)

        # Purge expired entries
        self.cache = [item for item in self.cache if (now - item["timestamp"]) < self.ttl_seconds]
        
        # Save disk updates if any expired items were removed
        if len(self.cache) != original_len:
            self._save_to_disk()

        if not self.cache:
            return None

        query_vector = np.array(self.embedder.embed_query(query))
        best_match = None
        highest_similarity = 0.0

        for item in self.cache:
            # 🔒 SECURITY CHECK: Skip user-private caches belonging to someone else
            if item["scope"] == "user" and item["user_id"] != current_user_id:
                continue

            sim = self._cosine_similarity(query_vector, item["vector"])
            if sim > highest_similarity:
                highest_similarity = sim
                best_match = item

        if highest_similarity >= self.similarity_threshold and best_match:
            logger.info(f"⚡ [CACHE HIT] ({best_match['scope'].upper()}) Match: '{best_match['query']}'")
            return best_match["response"]

        logger.info(f"🐢 [CACHE MISS] Query: '{query}'")
        return None

    def set(self, query: str, response: str, scope: str = "global", user_id: str = None):
        """Stores query response and persists to disk."""
        query_vector = np.array(self.embedder.embed_query(query))
        
        self.cache.append({
            "query": query,
            "vector": query_vector,
            "response": response,
            "timestamp": time.time(),
            "scope": scope,
            "user_id": user_id
        })

        # 🌟 Auto-save to disk immediately after inserting
        self._save_to_disk()
        logger.info(f"💾 [CACHE STORED] Scope: {scope.upper()} | User: {user_id} | Query: '{query}'")