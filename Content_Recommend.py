# RAG Content Recommender System using Chroma
# Part 1: Architecture and Dependencies

import json
from typing import List, Dict
import os
import warnings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import httpx
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

class RAGContentRecommender:
    """
    A recommendation system that combines vector similarity search with LLM-enhanced output generation.
    Uses Pinecone as the vector database with performance optimizations and load balancing.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-V2",  # Faster model
        groq_api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY") or os.environ.get("pinecone_api_key"),
        pinecone_index_name: str = "content-recommender",
        use_memory_cache: bool = True,
        descriptions_cache_path=os.environ.get("DESCRIPTION_PATH") or os.environ.get("description_path"),
        load_balancer=None  # Accept load balancer from main class
    ):
        # Set up embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.use_memory_cache = use_memory_cache
        self.descriptions_cache_path = descriptions_cache_path
        self.load_balancer = load_balancer
        
        # Performance optimization: Connection pooling
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Async HTTP client for better performance
        self._http_client = None
        self._init_async_client()

        if not pinecone_api_key:
            raise ValueError("Pinecone API key is required.")
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name

        # Set up Pinecone client
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            if self.pinecone_index_name not in self.pc.list_indexes().names():
                 raise RuntimeError(f"Pinecone index '{self.pinecone_index_name}' does not exist. "
                                    "Please create and populate it before using this system.")
            self.index = self.pc.Index(self.pinecone_index_name)
            print(f"Successfully connected to Pinecone index '{self.pinecone_index_name}'.")

        except Exception as e:
            raise RuntimeError(f"Could not initialize Pinecone: {e}") from e

        self.content_descriptions = {}
        self._load_content_descriptions()

        # Initialize models - use load balancer if available, otherwise fallback
        if self.load_balancer:
            # Use load balanced models for better performance and quality
            print("Content Recommender: Using load balanced models")
        else:
            # Fallback to direct model initialization
            self.groq2 = ChatGroq(
                model_name="llama3-8b-8192",  # Faster model
                api_key=groq_api_key,
                temperature=0.7,
                max_tokens=1000  # Reasonable limit
            )
            print("Content Recommender: Using fallback direct models")

    def _init_async_client(self):
        """Initialize async HTTP client with connection pooling"""
        try:
            # Use httpx for async HTTP operations with connection pooling
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                headers={"User-Agent": "MovieRecommender/1.0"}
            )
        except Exception as e:
            print(f"Failed to initialize async HTTP client: {e}")
            self._http_client = None

    def _load_content_descriptions(self):
        """Load content descriptions from disk if available."""
        if os.path.exists(self.descriptions_cache_path):
            try:
                with open(self.descriptions_cache_path, 'rb') as f:
                    self.content_descriptions = pickle.load(f)
                return True
            except Exception as e:
                print(f"Error loading content descriptions: {e}")
                self.content_descriptions = {}
                return False
        else:
            print("No saved content descriptions found.")
            self.content_descriptions = {}
            return False

    async def _analyze_query_context(self, query: str) -> Dict:
        """Use LLM to analyze query context and extract structured information."""
        prompt = """Analyze this content recommendation query and extract key information.
        Return a JSON object with these fields:
        - content_type: "movie", "tv", or "both"
        - genres: list of mentioned or implied genres
        - themes: list of themes or moods
        - time_period: temporal preference or null
        - similar_content: referenced titles for comparison
        - attributes: specific features requested
        - keywords: important query terms

        Query: {query}

        Respond ONLY with valid JSON:
        {{
            "content_type": "movie"|"tv"|"both",
            "genres": [],
            "themes": [],
            "time_period": string|null,
            "similar_content": [],
            "attributes": [],
            "keywords": []
        }}"""

        try:
            template = ChatPromptTemplate.from_template(prompt)
            
            # Use load balancer for quality analysis if available
            if self.load_balancer:
                # Query analysis is medium complexity - use quality models
                llm = self.load_balancer.get_model("quality")
            else:
                llm = self.groq2
                
            chain = LLMChain(llm=llm, prompt=template)
            response = await chain.ainvoke({"query": query})

            raw_text = response['text'].strip()

            if '```' in raw_text:
                code_block_parts = raw_text.split('```')
                if len(code_block_parts) >= 3:
                    raw_text = code_block_parts[1]
                    if raw_text.startswith('json'):
                        raw_text = raw_text[4:].strip()

            try:
                context = json.loads(raw_text)
                # Validate required fields
                required_fields = ['content_type', 'genres', 'themes', 'time_period',
                                'similar_content', 'attributes', 'keywords']
                for field in required_fields:
                    if field not in context:
                        raise KeyError(f"Missing required field: {field}")
                return context

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                raise ValueError("Invalid JSON response from LLM")

        except Exception as e:
            print(f"Error in LLM query analysis: {e}")
            # Fallback to basic analysis
            return {
                'content_type': 'movie' if 'movie' in query.lower() or 'film' in query.lower()
                            else 'tv' if any(word in query.lower() for word in ['tv', 'show', 'series'])
                            else 'both',
                'genres': [],
                'themes': [],
                'time_period': None,
                'similar_content': [],
                'attributes': ["directed by Christopher Nolan"] if "Christopher Nolan" in query else [],
                'keywords': [word for word in query.lower().split()
                            if word not in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}]
            }

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced search with optimized performance."""
        query_lower = query.lower()
        
        try:
            # Parallel execution of query analysis and embedding
            query_context_task = asyncio.create_task(self._analyze_query_context(query_lower))
            
            # Generate embedding in thread pool
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                self._executor, 
                self.embedding_model.encode,
                query
            )
            query_vector = query_vector.tolist()
            
            # Wait for query context analysis
            query_context = await query_context_task
            
            search_k = min(top_k * 3, 15)  # Reduced search size for speed

            # Async Pinecone query with better error handling
            try:
                results = await asyncio.to_thread(
                    self.index.query,
                    vector=query_vector,
                    top_k=search_k,
                    include_metadata=True
                )
            except Exception as e:
                print(f"Pinecone query error: {e}")
                # Fallback to smaller search if the query fails
                if search_k > 5:
                    results = await asyncio.to_thread(
                        self.index.query,
                        vector=query_vector,
                        top_k=5,
                        include_metadata=True
                    )
                else:
                    return []
        except Exception as e:
            print(f"Error during search: {e}")
            return []

        if not results or not results.get('matches'):
            return []

        all_items = []
        for match in results['matches']:
            content_id = match.get('id')
            metadata = match.get('metadata', {})
            score = match.get('score', 0.0)
            description = self.content_descriptions.get(content_id)

            try:
                content_data = json.loads(description)
                basic_info = content_data.get('basic_info', {})
                content_type = basic_info.get('type', '')
                title = basic_info.get('title', metadata.get('title', 'Unknown'))

                # Filter by content type if specified
                if query_context['content_type'] != 'both':
                    if (query_context['content_type'] == 'movie' and content_type != 'Movie') or \
                       (query_context['content_type'] == 'tv' and content_type != 'TV Show'):
                        continue

                # Extract additional metadata from content_data
                imdb_info = content_data.get('imdb_info', {})
                tmdb_info = content_data.get('tmdb_info', {})
                
                all_items.append({
                    'id': content_id,
                    'title': title,
                    'type': content_type,
                    'score': score,
                    'description': description,
                    'basic_info': basic_info,
                    'imdb_info': imdb_info,
                    'tmdb_info': tmdb_info,
                    'metadata': metadata
                })

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Skipping item {content_id} due to data error: {e}")
                continue

        all_items.sort(key=lambda x: x['score'], reverse=True)
        return all_items[:top_k]

    def _create_prompt(self, query: str, retrieved_content: List[Dict]) -> str:
        """Create an optimized prompt for recommendations."""
        query_lower = query.lower()
        if any(word in query_lower for word in ['movie', 'film']):
            content_type = "movies"
        elif any(word in query_lower for word in ['tv', 'show', 'series']):
            content_type = "tv series"
        else:
            content_type = "movies and tv series"

        prompt = f"""You are an expert media recommendation system.
        Based on the user's query and the JSON content information, recommend the most relevant content.
        For the query: {query}, Recommend the best {content_type} matching the context.
        
        RULES:
        - When asked for recent content, ONLY consider 2021-2025
        - Do not recommend the same title mentioned in the query
        - Limit to top 3 recommendations
        - Ensure recommendations match "{content_type}" type
        - For director queries, only include that director's work
        - Highlight unique aspects, not generic descriptions
        - Be concise and relevant

        Retrieved Content:
        """
        for i, content in enumerate(retrieved_content, 1):
            prompt += f"\n--- Item {i} ---\n{content['description']}\n"

        prompt += """
        Format your response as:
        ## Recommendations
        1. **[Title]** ([Type], [Year]) - [brief, specific reason why it matches the query]
        2. **[Title]** ([Type], [Year]) - [brief, specific reason why it matches the query]
        3. **[Title]** ([Type], [Year]) - [brief, specific reason why it matches the query]
        
        Make titles bold and keep descriptions concise but informative.
        """
        return prompt

    async def recommend(self, query: str, top_k: int = 3) -> str:
        """Generate optimized recommendations with load balanced models."""
        try:
            retrieved_content = await self.search(query, top_k=top_k*2)
            if not retrieved_content:
                return "No relevant content found. Please try a different query."

            prompt_text = self._create_prompt(query, retrieved_content)
            meta_prompt = ChatPromptTemplate.from_template("{prompt_text}")

            # Use load balancer for recommendation generation if available
            if self.load_balancer:
                # Recommendation generation is complex - use quality models (with LLaMA 4)
                llm = self.load_balancer.get_model("quality")
            else:
                llm = self.groq2

            chain = LLMChain(llm=llm, prompt=meta_prompt)
            response = await chain.ainvoke({"prompt_text": prompt_text})
            return response['text'].strip()
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return f"Error generating recommendations: {e}"
    
    async def recommend_with_metadata(self, query: str, top_k: int = 3) -> Dict:
        """Generate recommendations with metadata for creating cards."""
        try:
            retrieved_content = await self.search(query, top_k=top_k*2)
            if not retrieved_content:
                return {
                    "response": "I couldn't find any relevant content to recommend. Please try a different query.",
                    "recommendations": []
                }
            
            prompt_text = self._create_prompt(query, retrieved_content)
            meta_prompt = ChatPromptTemplate.from_template("{prompt_text}")

            # Use load balancer for recommendation generation if available
            if self.load_balancer:
                llm = self.load_balancer.get_model("quality")
            else:
                llm = self.groq2

            chain = LLMChain(llm=llm, prompt=meta_prompt)
            response = await chain.ainvoke({"prompt_text": prompt_text})
            
            # Format recommendation cards from the top results
            recommendation_cards = []
            for item in retrieved_content[:top_k]:
                # Try to extract TMDB/IMDB IDs from the data
                tmdb_id = None
                imdb_id = None
                year = None
                overview = None
                poster_path = None
                
                # Extract from basic_info first
                basic_info = item.get('basic_info', {})
                if basic_info:
                    year = basic_info.get('year')
                    overview = basic_info.get('overview') or basic_info.get('plot')
                
                # Extract from TMDB info
                tmdb_info = item.get('tmdb_info', {})
                if tmdb_info:
                    tmdb_id = tmdb_info.get('id') or tmdb_info.get('tmdb_id')
                    if not overview:
                        overview = tmdb_info.get('overview')
                    if not poster_path:
                        poster_path = tmdb_info.get('poster_path')
                    if not year and tmdb_info.get('release_date'):
                        try:
                            year = int(tmdb_info.get('release_date')[:4])
                        except:
                            pass
                
                # Extract from IMDB info
                imdb_info = item.get('imdb_info', {})
                if imdb_info:
                    imdb_id = imdb_info.get('imdb_id') or imdb_info.get('id')
                    if not year and imdb_info.get('year'):
                        year = imdb_info.get('year')
                
                # Create card data structure similar to TMDB format
                card_data = {
                    'id': tmdb_id or item.get('id'),
                    'title': item.get('title'),
                    'name': item.get('title') if item.get('type') == 'TV Show' else None,
                    'overview': overview or 'No overview available.',
                    'poster_path': poster_path,
                    'release_date': f"{year}-01-01" if year else None,
                    'first_air_date': f"{year}-01-01" if year and item.get('type') == 'TV Show' else None,
                    'media_type': 'tv' if item.get('type') == 'TV Show' else 'movie',
                    'vote_average': 0,
                    'popularity': item.get('score', 0) * 100,
                    # Additional metadata
                    'imdb_id': imdb_id,
                    'content_score': item.get('score', 0)
                }
                
                recommendation_cards.append(card_data)
            
            return {
                "response": response['text'].strip(),
                "recommendations": recommendation_cards
            }
            
        except Exception as e:
            print(f"Error generating recommendations with metadata: {e}")
            return {
                "response": f"Error generating recommendations: {e}",
                "recommendations": []
            }

    # Add alias for async method
    async def arecommend(self, query: str, top_k: int = 3) -> str:
        """Alias for async recommend method."""
        return await self.recommend(query, top_k)





# async def main():

#     recommender = RAGContentRecommender()

#     queries = [
#         # "Shows like Breaking Bad with a complex protagonist",
#         # "Movies directed by Christopher Nolan",
#         "which are the top hbo shows",
#         "top paramount movies",
#         # "Feel-good comedies of male leads",
#         # "Science fiction TV shows with time travel",
#         # "Recent action movies with high ratings",
#         # "movies with great cgi like dune part two",
#         # "japanese anime like bleach",
#         # "movies like interstellar",
#         # "recommend best marvel movies",
#         # "best batman movies",
#         # "best dc movies"

#     ]

#     # Process each query
#     for i, query in enumerate(queries, 1):
#         print(f"\n--- Query {i}: {query} ---")

#         # Get recommendations
#         recommendations = await recommender.recommend(query, top_k=5)

#         # Print recommendations
#         print(recommendations)
#         print("\n" + "-"*80)

# if __name__ == "__main__":
#     asyncio.run(main())