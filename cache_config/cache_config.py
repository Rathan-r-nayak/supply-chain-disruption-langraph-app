from cache_config.semantic_cache import LocalSemanticCache
from Config.llm_config import embedding_model

app_semantic_cache = LocalSemanticCache(
    embedding_model=embedding_model, # Reusing your Azure embeddings
    ttl_seconds=3600, 
    similarity_threshold=0.92,
    cache_file="cache_config/semantic_cache.pkl"
)