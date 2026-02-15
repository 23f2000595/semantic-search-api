# main.py - Complete Semantic Search with Re-ranking

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import time
import openai
import os
import json
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Semantic Search API", description="Search with re-ranking for product reviews")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Using mock embeddings for testing.")
    openai.api_key = "mock_key"
else:
    openai.api_key = OPENAI_API_KEY

# ==================== DATA ====================
# 76 customer reviews (product reviews domain)
REVIEWS = [
    {"id": 0, "content": "Battery life is amazing! Lasts whole day with heavy use.", "metadata": {"product": "Laptop X1", "rating": 5}},
    {"id": 1, "content": "Poor battery performance, drains quickly even on standby.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 2, "content": "Camera quality is outstanding, especially in low light.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 3, "content": "Screen is too dim outdoors, barely visible in sunlight.", "metadata": {"product": "Laptop X1", "rating": 3}},
    {"id": 4, "content": "Fast charging works great, 0 to 100 in 30 minutes.", "metadata": {"product": "Charger Z3", "rating": 4}},
    {"id": 5, "content": "Very disappointed with customer support, no response for weeks.", "metadata": {"product": "Service", "rating": 1}},
    {"id": 6, "content": "Build quality is premium, feels solid in hand.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 7, "content": "Software is buggy, frequent crashes and lags.", "metadata": {"product": "Laptop X1", "rating": 2}},
    {"id": 8, "content": "Excellent value for money, performs beyond expectations.", "metadata": {"product": "Tablet A4", "rating": 5}},
    {"id": 9, "content": "Heating issues during gaming, becomes uncomfortable to hold.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 10, "content": "Speakers are loud and clear, great for music.", "metadata": {"product": "Laptop X1", "rating": 4}},
    {"id": 11, "content": "WiFi connectivity is weak, keeps disconnecting.", "metadata": {"product": "Router R1", "rating": 2}},
    {"id": 12, "content": "Setup was easy, instructions were clear.", "metadata": {"product": "Router R1", "rating": 4}},
    {"id": 13, "content": "Very heavy to carry around, not portable at all.", "metadata": {"product": "Laptop X1", "rating": 3}},
    {"id": 14, "content": "Touchscreen is responsive and accurate.", "metadata": {"product": "Tablet A4", "rating": 5}},
    {"id": 15, "content": "Battery swelled after 6 months, dangerous!", "metadata": {"product": "Phone Y2", "rating": 1}},
    {"id": 16, "content": "Love the design, very sleek and modern.", "metadata": {"product": "Laptop X1", "rating": 5}},
    {"id": 17, "content": "App crashes every time I try to open settings.", "metadata": {"product": "Software", "rating": 1}},
    {"id": 18, "content": "Fast and smooth performance, no lag at all.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 19, "content": "Charging port is loose, cable falls out easily.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 20, "content": "Fingerprint sensor is fast and accurate.", "metadata": {"product": "Laptop X1", "rating": 4}},
    {"id": 21, "content": "Bluetooth keeps disconnecting from my headphones.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 22, "content": "Display colors are vibrant and lifelike.", "metadata": {"product": "Tablet A4", "rating": 5}},
    {"id": 23, "content": "Fan noise is too loud even on light tasks.", "metadata": {"product": "Laptop X1", "rating": 2}},
    {"id": 24, "content": "Delivery was delayed by 2 weeks, very frustrating.", "metadata": {"product": "Service", "rating": 1}},
    {"id": 25, "content": "Water resistance works well, dropped in pool and survived.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 26, "content": "Keyboard is comfortable for long typing sessions.", "metadata": {"product": "Laptop X1", "rating": 4}},
    {"id": 27, "content": "Battery backup is excellent, lasts 2 days on single charge.", "metadata": {"product": "Tablet A4", "rating": 5}},
    {"id": 28, "content": "Software update made phone slower, regret updating.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 29, "content": "Great customer service, resolved issue immediately.", "metadata": {"product": "Service", "rating": 5}},
    {"id": 30, "content": "Overpriced for the features it offers.", "metadata": {"product": "Laptop X1", "rating": 2}},
    {"id": 31, "content": "Face unlock works even in dark conditions.", "metadata": {"product": "Phone Y2", "rating": 4}},
    {"id": 32, "content": "Hinge is loose after few months of use.", "metadata": {"product": "Laptop X1", "rating": 2}},
    {"id": 33, "content": "Fast charging ruined my battery health within months.", "metadata": {"product": "Phone Y2", "rating": 1}},
    {"id": 34, "content": "Great for gaming, runs all games smoothly.", "metadata": {"product": "Laptop X1", "rating": 5}},
    {"id": 35, "content": "Poor call quality, voice echoes on other side.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 36, "content": "Lightweight and portable, easy to carry.", "metadata": {"product": "Tablet A4", "rating": 4}},
    {"id": 37, "content": "USB ports are too tight, hard to connect devices.", "metadata": {"product": "Laptop X1", "rating": 3}},
    {"id": 38, "content": "Excellent camera, photos look professional.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 39, "content": "Warranty claim was rejected for no reason.", "metadata": {"product": "Service", "rating": 1}},
    {"id": 40, "content": "Battery drains fast when using GPS.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 41, "content": "Sleek design, looks premium and expensive.", "metadata": {"product": "Laptop X1", "rating": 4}},
    {"id": 42, "content": "Charger stopped working after 1 week.", "metadata": {"product": "Charger Z3", "rating": 1}},
    {"id": 43, "content": "Display is too reflective, glare is annoying.", "metadata": {"product": "Tablet A4", "rating": 3}},
    {"id": 44, "content": "Good battery life for normal usage.", "metadata": {"product": "Phone Y2", "rating": 4}},
    {"id": 45, "content": "Frequent software bugs, needs restart daily.", "metadata": {"product": "Router R1", "rating": 2}},
    {"id": 46, "content": "Amazing sound quality, best in class.", "metadata": {"product": "Laptop X1", "rating": 5}},
    {"id": 47, "content": "Scratches easily, not durable at all.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 48, "content": "Fast delivery, well packaged.", "metadata": {"product": "Service", "rating": 5}},
    {"id": 49, "content": "Heating issue fixed after software update.", "metadata": {"product": "Phone Y2", "rating": 4}},
    {"id": 50, "content": "Poor build quality, creaking sounds.", "metadata": {"product": "Laptop X1", "rating": 2}},
    {"id": 51, "content": "Battery health dropped to 80% in 3 months.", "metadata": {"product": "Phone Y2", "rating": 1}},
    {"id": 52, "content": "Bright display, good for outdoor use.", "metadata": {"product": "Tablet A4", "rating": 4}},
    {"id": 53, "content": "Worth every penny, highly recommend.", "metadata": {"product": "Laptop X1", "rating": 5}},
    {"id": 54, "content": "Update improved battery life significantly.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 55, "content": "Cheap plastic feel, not premium.", "metadata": {"product": "Router R1", "rating": 2}},
    {"id": 56, "content": "Responsive touch, no input lag.", "metadata": {"product": "Tablet A4", "rating": 4}},
    {"id": 57, "content": "Never buying this brand again.", "metadata": {"product": "Phone Y2", "rating": 1}},
    {"id": 58, "content": "Good for basic tasks, not for heavy work.", "metadata": {"product": "Laptop X1", "rating": 3}},
    {"id": 59, "content": "Battery replacement was expensive.", "metadata": {"product": "Service", "rating": 2}},
    {"id": 60, "content": "Quick charges, full in 20 minutes.", "metadata": {"product": "Charger Z3", "rating": 5}},
    {"id": 61, "content": "Fingerprint magnet, looks dirty always.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 62, "content": "Perfect size for reading ebooks.", "metadata": {"product": "Tablet A4", "rating": 5}},
    {"id": 63, "content": "Customer support was rude and unhelpful.", "metadata": {"product": "Service", "rating": 1}},
    {"id": 64, "content": "Long lasting battery, impressed.", "metadata": {"product": "Laptop X1", "rating": 5}},
    {"id": 65, "content": "Camera glass broke without dropping.", "metadata": {"product": "Phone Y2", "rating": 1}},
    {"id": 66, "content": "Excellent battery backup, 8 hours SOT.", "metadata": {"product": "Phone Y2", "rating": 5}},
    {"id": 67, "content": "Too expensive for what it offers.", "metadata": {"product": "Tablet A4", "rating": 2}},
    {"id": 68, "content": "No issues after 6 months of use.", "metadata": {"product": "Laptop X1", "rating": 4}},
    {"id": 69, "content": "Charging is very slow, takes hours.", "metadata": {"product": "Phone Y2", "rating": 2}},
    {"id": 70, "content": "Best purchase I made this year.", "metadata": {"product": "Tablet A4", "rating": 5}},
    {"id": 71, "content": "Wifi range is excellent, covers whole house.", "metadata": {"product": "Router R1", "rating": 5}},
    {"id": 72, "content": "Battery drains overnight even when idle.", "metadata": {"product": "Phone Y2", "rating": 1}},
    {"id": 73, "content": "Solid performance for the price.", "metadata": {"product": "Laptop X1", "rating": 4}},
    {"id": 74, "content": "Software is intuitive and easy to use.", "metadata": {"product": "Tablet A4", "rating": 4}},
    {"id": 75, "content": "Would not recommend to anyone.", "metadata": {"product": "Phone Y2", "rating": 1}},
]

# ==================== MODELS ====================
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6

class SearchResult(BaseModel):
    id: int
    score: float
    content: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    reranked: bool
    metrics: Dict[str, Any]

# ==================== EMBEDDING CACHE ====================
class EmbeddingCache:
    def __init__(self):
        self.document_embeddings = None
        self.documents = REVIEWS
        
    def compute_embeddings(self):
        """Compute embeddings for all documents"""
        if self.document_embeddings is not None:
            return self.document_embeddings
            
        logger.info("Computing embeddings for 76 documents...")
        document_texts = [doc['content'] for doc in self.documents]
        
        # Use mock embeddings if no API key
        if OPENAI_API_KEY == "mock_key":
            np.random.seed(42)
            self.document_embeddings = np.random.randn(len(document_texts), 1536)
            # Normalize for cosine similarity
            self.document_embeddings = self.document_embeddings / np.linalg.norm(self.document_embeddings, axis=1, keepdims=True)
        else:
            self.document_embeddings = self.get_embeddings_batch(document_texts)
        
        logger.info(f"Embeddings computed: {self.document_embeddings.shape}")
        return self.document_embeddings
    
    def get_embeddings_batch(self, texts, batch_size=20):
        """Get embeddings in batches to avoid rate limits"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = openai.Embedding.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error getting embeddings: {e}")
                # Fallback to random embeddings
                np.random.seed(i)
                fallback = np.random.randn(len(batch), 1536)
                fallback = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
                all_embeddings.extend(fallback)
        
        return np.array(all_embeddings)
    
    def get_query_embedding(self, query):
        """Get embedding for a single query"""
        if OPENAI_API_KEY == "mock_key":
            np.random.seed(hash(query) % 2**32)
            embedding = np.random.randn(1536)
            return embedding / np.linalg.norm(embedding)
        
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=query
            )
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            np.random.seed(hash(query) % 2**32)
            embedding = np.random.randn(1536)
            return embedding / np.linalg.norm(embedding)

# Initialize cache
cache = EmbeddingCache()

# ==================== SEARCH FUNCTIONS ====================
def initial_retrieval(query_embedding, k=10):
    """Stage 1: Fast vector search to get top k candidates"""
    # Get document embeddings from cache
    doc_embeddings = cache.compute_embeddings()
    
    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Prepare candidates with initial scores
    candidates = []
    for idx in top_indices:
        candidates.append({
            'id': cache.documents[idx]['id'],
            'content': cache.documents[idx]['content'],
            'metadata': cache.documents[idx]['metadata'],
            'initial_score': float(similarities[idx])
        })
    
    logger.info(f"Initial retrieval: found {len(candidates)} candidates")
    return candidates

def rerank_with_llm(query, candidates, rerankK=6):
    """Stage 2: Re-rank candidates using LLM-based relevance scoring"""
    
    logger.info(f"Re-ranking {len(candidates)} candidates with LLM")
    
    # If using mock, simulate re-ranking scores
    if OPENAI_API_KEY == "mock_key":
        for i, candidate in enumerate(candidates):
            # Simulate relevance based on keyword matching
            query_words = set(query.lower().split())
            doc_words = set(candidate['content'].lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = min(overlap / max(len(query_words), 1), 1.0) * 10
            candidate['rerank_score'] = score / 10.0
    else:
        # Batch process with actual LLM
        for candidate in candidates:
            prompt = f"""Query: {query}
Document: {candidate['content']}

Rate the relevance of this document to the query on a scale of 0-10.
Consider: How well does this document address the query?
Respond with only the number (0-10)."""

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                score_text = response.choices[0].message.content.strip()
                # Extract number from response
                import re
                numbers = re.findall(r'\d+\.?\d*', score_text)
                if numbers:
                    score = float(numbers[0])
                else:
                    score = 5.0
                
                # Clamp between 0-10
                score = max(0, min(10, score))
                candidate['rerank_score'] = score / 10.0
                
            except Exception as e:
                logger.error(f"LLM scoring error: {e}")
                candidate['rerank_score'] = candidate['initial_score']
    
    # Sort by rerank score
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Handle ties: if scores are equal, preserve initial retrieval order
    for i in range(len(candidates)-1):
        if abs(candidates[i]['rerank_score'] - candidates[i+1]['rerank_score']) < 0.001:
            # If tie and initial order is different, sort by initial score
            if candidates[i]['initial_score'] < candidates[i+1]['initial_score']:
                candidates[i], candidates[i+1] = candidates[i+1], candidates[i]
    
    return candidates[:rerankK]

# ==================== API ENDPOINTS ====================
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Main search endpoint with two-stage retrieval and re-ranking
    """
    start_time = time.time()
    
    # Validate input
    if request.k < 1 or request.rerankK < 1:
        raise HTTPException(status_code=400, detail="k and rerankK must be positive")
    
    if request.rerankK > request.k:
        logger.warning("rerankK > k, using k as limit")
        request.rerankK = request.k
    
    try:
        # Get query embedding
        query_embedding = cache.get_query_embedding(request.query)
        
        # Stage 1: Initial retrieval
        candidates = initial_retrieval(query_embedding, request.k)
        
        # Handle no results
        if not candidates:
            return SearchResponse(
                results=[],
                reranked=request.rerank,
                metrics={
                    "latency": int((time.time() - start_time) * 1000),
                    "totalDocs": len(cache.documents)
                }
            )
        
        # Stage 2: Re-ranking (if enabled)
        if request.rerank:
            final_results = rerank_with_llm(request.query, candidates, request.rerankK)
            reranked = True
        else:
            # Just take top rerankK from initial retrieval
            final_results = candidates[:request.rerankK]
            reranked = False
        
        # Format results with proper scores
        results = []
        for r in final_results:
            score = r.get('rerank_score', r.get('initial_score', 0))
            # Ensure score is between 0-1
            score = max(0, min(1, score))
            
            results.append(SearchResult(
                id=r['id'],
                score=round(score, 3),
                content=r['content'],
                metadata=r['metadata']
            ))
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Ensure latency requirement (<200ms for initial retrieval)
        if not request.rerank and latency_ms > 200:
            logger.warning(f"Initial retrieval latency {latency_ms}ms > 200ms")
        
        return SearchResponse(
            results=results,
            reranked=reranked,
            metrics={
                "latency": latency_ms,
                "totalDocs": len(cache.documents),
                "initialCandidates": len(candidates),
                "finalResults": len(results)
            }
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "total_docs": len(cache.documents),
        "embeddings_cached": cache.document_embeddings is not None,
        "api_key_configured": OPENAI_API_KEY != "mock_key"
    }

@app.get("/docs/{doc_id}")
async def get_document(doc_id: int):
    """Get a specific document by ID"""
    if 0 <= doc_id < len(cache.documents):
        return cache.documents[doc_id]
    raise HTTPException(status_code=404, detail="Document not found")

# ==================== EVALUATION ====================
@app.post("/evaluate")
async def evaluate_precision():
    """
    Simple evaluation to check precision@6 > 0.65
    """
    test_queries = [
        "battery life issues",
        "camera quality",
        "customer support",
        "performance lag",
        "display problems",
        "charging speed"
    ]
    
    expected_relevant = {
        "battery life issues": [0, 1, 15, 27, 33, 40, 44, 51, 54, 66, 72],
        "camera quality": [2, 38, 65],
        "customer support": [5, 29, 39, 48, 63],
        "performance lag": [7, 17, 18, 28, 45, 73],
        "display problems": [3, 22, 43, 52],
        "charging speed": [4, 42, 60, 69]
    }
    
    results = {}
    total_precision = 0
    
    for query in test_queries:
        # Get query embedding
        query_embedding = cache.get_query_embedding(query)
        
        # Initial retrieval
        candidates = initial_retrieval(query_embedding, 10)
        
        # Re-rank
        reranked = rerank_with_llm(query, candidates, 6)
        
        # Calculate precision@6
        relevant_ids = expected_relevant.get(query, [])
        if relevant_ids:
            retrieved_ids = [r['id'] for r in reranked]
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision = relevant_retrieved / 6
            total_precision += precision
            results[query] = {
                "precision@6": precision,
                "retrieved": retrieved_ids,
                "relevant": relevant_ids
            }
    
    avg_precision = total_precision / len(test_queries)
    
    return {
        "average_precision@6": avg_precision,
        "meets_requirement": avg_precision > 0.65,
        "detailed_results": results
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
