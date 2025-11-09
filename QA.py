import os
import pickle
import re
import string
import warnings
import time
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

class MovieQASystem:
    """
    Optimized question-answering system that combines vector similarity search with LLM-enhanced
    responses about movies and TV shows. Uses Pinecone as the vector database with performance optimizations
    and load balancing.
    """

    def __init__(
        self,
        csv_path=os.environ.get("PLOTS_PATH") or os.environ.get("plots_path"),
        additional_data_path=os.environ.get("DESCRIPTION_PATH") or os.environ.get("description_path"),
        embedding_model_name: str = "all-MiniLM-L6-V2",  # Already using fast model
        google_api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("google_api_key"),
        groq_api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY") or os.environ.get("pinecone_api_key"),
        pinecone_index_name: str = "movie-plot",
        use_memory_cache: bool = True,
        metadata_cache_path=os.environ.get("METADATA_PATH") or os.environ.get("metadata_path"),
        load_balancer=None  # Accept load balancer from main class
    ):
        self.additional_data_path = additional_data_path
        self.use_memory_cache = use_memory_cache
        self.metadata_cache_path = metadata_cache_path
        self.csv_path = csv_path
        self.pickle_data = None
        self.title_to_imdb = {}
        self.load_balancer = load_balancer
        
        # Performance optimizations: Enhanced caching system
        self._query_cache: Dict[str, Tuple[str, float]] = {}  # (response, timestamp)
        self._title_cache: Dict[str, Tuple[str, str]] = {}  # (title, clean_query)
        self._movie_data_cache: Dict[str, Tuple] = {}  # movie data by title
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        # Connection pooling for better performance
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Common movie title patterns for smart bypass
        self._title_indicators = {
            'movie', 'film', 'the movie', 'the film', 'in the movie', 'in the film',
            'about the movie', 'about the film', 'plot of', 'story of', 'ending of'
        }

        self.embedding_model = SentenceTransformer(embedding_model_name)

        if not pinecone_api_key:
            raise ValueError("Pinecone API key is required.")
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name

        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            if self.pinecone_index_name not in self.pc.list_indexes().names():
                 raise RuntimeError(f"Pinecone index '{self.pinecone_index_name}' does not exist. "
                                    "Please create and populate it before using this system.")
            self.index = self.pc.Index(self.pinecone_index_name)
            print(f"Successfully connected to Pinecone index '{self.pinecone_index_name}'.")

        except Exception as e:
            raise RuntimeError(f"Could not initialize Pinecone: {e}") from e

        self._load_metadata()
        self.load_pickle_data()
        
        # Load DataFrame only if needed and cache it
        self.plots_df = None
        if os.path.exists(self.csv_path):
            self.plots_df = pd.read_csv(self.csv_path)
        
        # Initialize models - use load balancer if available, otherwise fallback
        if self.load_balancer:
            print("QA System: Using load balanced models")
        else:
            # Fallback to direct model initialization
            self.groq = ChatGroq(
                model_name="llama3-8b-8192",  # Much faster model
                api_key=groq_api_key,
                temperature=0.2,
                max_tokens=100  # Limit for title extraction
            )
            self.groq1 = ChatGroq(
                model_name="llama3-8b-8192",  # Consistent fast model
                api_key=groq_api_key,
                temperature=0.3,
                max_tokens=1000  # Reasonable limit for responses
            )
            self.gemini = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=google_api_key,
                temperature=0.4
            )
            print("QA System: Using fallback direct models")

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self._cache_ttl

    def _clean_expired_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        
        # Clean query cache
        expired_keys = [
            key for key, (_, timestamp) in self._query_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._query_cache[key]
        
        # Clean other caches similarly
        expired_title_keys = [
            key for key, (_, timestamp) in self._title_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_title_keys:
            del self._title_cache[key]

    def _smart_title_extraction(self, query: str) -> Optional[str]:
        """Fast pattern-based title extraction for obvious cases"""
        query_lower = query.lower()
        
        # Look for quoted titles
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted_matches:
            return quoted_matches[0]
        
        # Look for patterns like "in the movie X" or "plot of X"
        patterns = [
            r'(?:in the (?:movie|film)|plot of|story of|about the (?:movie|film)) ([a-zA-Z0-9\s:]+?)(?:\s|$|,|\?)',
            r'(?:movie|film) ([a-zA-Z0-9\s:]+?)(?:\s|$|,|\?)',
            r'([a-zA-Z0-9\s:]{2,30})(?:\s(?:movie|film)|\'s plot|\'s story)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                title = matches[0].strip()
                if len(title) > 2 and len(title) < 50:  # Reasonable title length
                    return title
        
        return None

    def _load_metadata(self):
        """Load metadata from disk if available."""
        if os.path.exists(self.metadata_cache_path):
            try:
                with open(self.metadata_cache_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.title_to_imdb = metadata.get('title_to_imdb', {})
                return True
            except Exception as e:
                self.title_to_imdb = {}
                return False
        else:
            self.title_to_imdb = {}
            return False

    def verify_pinecone_collection(self):
        """Verify the existing Pinecone collection and its contents."""
        try:
            stats = self.index.describe_index_stats()
            if stats.total_vector_count > 0:
                return True
            else:
                print(f"Warning: Pinecone index '{self.pinecone_index_name}' appears to be empty.")
                return False
        except Exception as e:
            print(f"Error verifying Pinecone index: {e}")
            return False

    def load_pickle_data(self, pickle_path=None):
        """Load additional movie data from pickle file"""
        path_to_use = pickle_path or self.additional_data_path

        try:
            with open(path_to_use, 'rb') as f:
                self.pickle_data = pickle.load(f)
        except Exception as e:
            self.pickle_data = {}

    async def find_best_title_match(self, query: str, top_k: int = 3):
        """Optimized title matching with caching"""
        cache_key = f"title_match_{query}_{top_k}"
        if cache_key in self._query_cache:
            cached_result, timestamp = self._query_cache[cache_key]
            if self._is_cache_valid(timestamp):
                return cached_result

        if not hasattr(self, 'index') or self.index is None:
            print("Error: Pinecone index not initialized.")
            return []

        try:
            # Use thread pool for embedding generation
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                self._executor,
                self.embedding_model.encode,
                query
            )
            query_vector = query_vector.tolist()
            
            # Pinecone query
            results = await asyncio.to_thread(
                self.index.query,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error during Pinecone query: {e}")
            return []

        matches = []
        if results and results.get('matches'):
            for match in results['matches']:
                metadata = match.get('metadata', {})
                title = metadata.get('title', 'Unknown Title')
                similarity = match.get('score', 0.0)
                similarity = max(0.0, min(1.0, similarity))
                matches.append((title, metadata, similarity))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Cache the result
        self._query_cache[cache_key] = (matches, time.time())
        return matches

    def _normalize_title(self, title: str) -> str:
        """Normalize a title string by converting to lowercase and removing punctuation."""
        if not title:
            return ""
        title = title.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(title.lower().split())

    async def get_movie_data_by_title(self, title_query):
        """Optimized movie data retrieval with caching"""
        # Check cache first
        if title_query in self._movie_data_cache:
            cached_result, timestamp = self._movie_data_cache[title_query]
            if self._is_cache_valid(timestamp):
                return cached_result

        if not hasattr(self, 'index') or self.index is None:
            return None, None, None
                            
        # 1. Try a normalized match using top vector search results
        try:
            norm_query_title = self._normalize_title(title_query)
            potential_matches = await self.find_best_title_match(title_query, top_k=3)  # Reduced for speed

            for p_title, p_metadata, p_similarity in potential_matches:
                # Check for exact normalized match first
                if self._normalize_title(p_title) == norm_query_title:
                    plot = p_metadata.get('plot')
                    wiki_intro = p_metadata.get('wikipedia_intro')
                    imdb_id = p_metadata.get('imdb_id')
                    if plot or wiki_intro:
                        result = (plot, wiki_intro, imdb_id)
                        # Cache the result
                        self._movie_data_cache[title_query] = (result, time.time())
                        return result
        except Exception as e:
            pass

        # 2. Fallback to DataFrame search if available
        if self.plots_df is not None and not self.plots_df.empty:
            try:
                df_matches = self.plots_df[self.plots_df['title'].str.lower() == title_query.lower()]
                if not df_matches.empty:
                    row = df_matches.iloc[0]
                    plot = row.get('plot')
                    wiki_intro = row.get('wikipedia_intro')
                    imdb_id = row.get('imdb_id')
                    if plot or wiki_intro:
                        result = (plot, wiki_intro, imdb_id)
                        # Cache the result
                        self._movie_data_cache[title_query] = (result, time.time())
                        return result
            except (KeyError, Exception) as e:
                pass

        # 3. Best similarity match as fallback
        if 'potential_matches' in locals() and potential_matches and potential_matches[0][2] > 0.7:
            best_match_meta = potential_matches[0][1]
            plot = best_match_meta.get('plot')
            wiki_intro = best_match_meta.get('wikipedia_intro')
            imdb_id = best_match_meta.get('imdb_id')
            if plot or wiki_intro:
                result = (plot, wiki_intro, imdb_id)
                # Cache the result
                self._movie_data_cache[title_query] = (result, time.time())
                return result

        # Cache negative result briefly
        result = (None, None, None)
        self._movie_data_cache[title_query] = (result, time.time())
        return result

    def get_movie_info_by_imdb_id(self, imdb_id):
        """Get additional movie information from pickle data by IMDB ID"""
        return self.pickle_data.get(imdb_id, None)

    async def _extract_title_with_llm(self, query, llm_name):
        """Optimized LLM title extraction with simpler prompt and load balancing"""
        # Try to get LLM from load balancer first
        if self.load_balancer:
            # Title extraction is simple task - use fast models
            llm = self.load_balancer.get_model("fast")
        else:
            # Fallback to instance attribute
            llm = getattr(self, llm_name) if hasattr(self, llm_name) else None
            
        if not llm:
            return None

        # Simplified prompt for faster processing
        title_prompt = ChatPromptTemplate.from_template("""
        Extract the movie/TV title from this query. Return ONLY the title, nothing else.
        If no title found, return "None".

        Query: "{query}"
        Title:
        """)

        try:
            chain = LLMChain(llm=llm, prompt=title_prompt)
            response = await chain.ainvoke({"query": query})
            extracted_title = response['text'].strip()

            if extracted_title.lower() == 'none':
                return None

            return extracted_title
        except Exception as e:
            print(f"Error in LLM title extraction: {e}")
            return None

    async def extract_title_from_query(self, query):
        """Optimized title extraction with smart bypass and caching"""
        # Check cache first
        if query in self._title_cache:
            cached_result, timestamp = self._title_cache[query]
            if self._is_cache_valid(timestamp):
                return cached_result

        # Try smart pattern-based extraction first
        smart_title = self._smart_title_extraction(query)
        if smart_title:
            # Validate if this title exists in our DB
            movie_data = await self.get_movie_data_by_title(smart_title)
            plot, wiki_intro, _ = movie_data            
            if plot or wiki_intro:
                try:
                    pattern = re.compile(r'\b' + re.escape(smart_title) + r'\b', re.IGNORECASE)
                    clean_query = pattern.sub('', query, count=1).strip()
                    clean_query = ' '.join(clean_query.split()) if clean_query != query else query
                except re.error:
                    clean_query = query
                
                result = (smart_title, clean_query if clean_query else query)
                # Cache the result
                self._title_cache[query] = (result, time.time())
                return result

        # Fallback to LLM extraction with load balancing
        extracted_title = None
        
        # Try with load balanced fast models first for speed
        if self.load_balancer:
            extracted_title = await self._extract_title_with_llm(query, 'fast')
        
        # Fallback to instance models if no load balancer
        if not extracted_title and hasattr(self, 'groq'):
            extracted_title = await self._extract_title_with_llm(query, 'groq')
        
        if not extracted_title and hasattr(self, 'gemini'):
            extracted_title = await self._extract_title_with_llm(query, 'gemini')

        if extracted_title:
            # Validate if this title exists in our DB
            movie_data = await self.get_movie_data_by_title(extracted_title)
            plot, wiki_intro, _ = movie_data            
            if plot or wiki_intro:
                try:
                    pattern = re.compile(r'\b' + re.escape(extracted_title) + r'\b', re.IGNORECASE)
                    clean_query = pattern.sub('', query, count=1).strip()
                    clean_query = ' '.join(clean_query.split()) if clean_query != query else query
                except re.error:
                    clean_query = query
                
                result = (extracted_title, clean_query if clean_query else query)
                # Cache the result
                self._title_cache[query] = (result, time.time())
                return result

        # Fallback to vector search
        matches = await self.find_best_title_match(query, top_k=3)
        if matches:
            best_match_title, _, best_similarity = matches[0]
            if best_similarity > 0.65:
                plot_vs, wiki_intro_vs, _ = await self.get_movie_data_by_title(best_match_title)
                if plot_vs or wiki_intro_vs:
                    try:
                        pattern = re.compile(r'\b' + re.escape(best_match_title) + r'\b', re.IGNORECASE)
                        clean_query = pattern.sub('', query, count=1).strip()
                        clean_query = ' '.join(clean_query.split()) if clean_query != query else query
                    except re.error:
                        clean_query = query

                    result = (best_match_title, clean_query if clean_query else query)
                    # Cache the result
                    self._title_cache[query] = (result, time.time())
                    return result

        # Cache negative result
        result = (None, query)
        self._title_cache[query] = (result, time.time())
        return result

    async def answer_query(self, query):
        """Optimized query answering with caching, performance improvements, and load balancing"""
        # Clean expired cache periodically
        self._clean_expired_cache()
        
        # Check cache first
        if query in self._query_cache:
            cached_response, timestamp = self._query_cache[query]
            if self._is_cache_valid(timestamp):
                return cached_response

        start_time = time.time()
        
        title, actual_query = await self.extract_title_from_query(query)
        if not title:
            matches = await self.find_best_title_match(query, top_k=3)

            if matches and matches[0][2] > 0.75: 
                title = matches[0][0]
                metadata = matches[0][1]
                actual_query = query
            else:
                response = "I couldn't identify a movie in your question. Please ensure you include the movie name or specific details about the plot."
                # Cache negative response briefly
                self._query_cache[query] = (response, time.time())
                return response

        movie_data = await self.get_movie_data_by_title(title)
        plot, wiki_intro, imdb_id = movie_data
        if not plot and not wiki_intro:
            response = f"I couldn't find information about '{title}'. Please check the movie title."
            # Cache negative response
            self._query_cache[query] = (response, time.time())
            return response

        additional_info = None
        if imdb_id:
            additional_info = self.get_movie_info_by_imdb_id(imdb_id)

        # Build context more efficiently
        context_lines = [f"Movie: {title}"]

        if plot and isinstance(plot, str) and plot.strip():
            context_lines.append(f"Plot: {plot}")

        if wiki_intro and isinstance(wiki_intro, str) and wiki_intro.strip():
            context_lines.append(f"Wikipedia Introduction: {wiki_intro}")

        if additional_info:
            basic_info = additional_info.get('basic_info', {})
            content = additional_info.get('content', {})
            people = additional_info.get('people', {})

            context_lines.append("")
            context_lines.append("Additional Details:")
            if basic_info.get('type'):
                context_lines.append(f"- Type: {basic_info.get('type')}")
            if basic_info.get('rating'):
                context_lines.append(f"- Rating: {basic_info.get('rating')}")
            if content.get('genres'):
                genres = content['genres']
                if isinstance(genres, (list, tuple)):
                    context_lines.append(f"- Genres: {', '.join(genres)}")
                else:
                    context_lines.append(f"- Genres: {genres}")
            if people.get('cast'):
                cast = people['cast']
                if isinstance(cast, (list, tuple)):
                    context_lines.append(f"- Main Cast: {', '.join(cast[:5])}")
                else:
                    context_lines.append(f"- Main Cast: {cast}")
            
            # Add movie/TV specific info more efficiently
            if 'movie_specific' in additional_info:
                movie_info = additional_info['movie_specific']
                if movie_info.get('release_date'):
                    context_lines.append(f"- Release Date: {movie_info['release_date']}")
                if movie_info.get('directors'):
                    directors = movie_info['directors']
                    if isinstance(directors, (list, tuple)):
                        context_lines.append(f"- Director(s): {', '.join(directors)}")
                    else:
                        context_lines.append(f"- Director(s): {directors}")
            elif 'tv_specific' in additional_info:
                tv_info = additional_info['tv_specific']
                if tv_info.get('seasons_count'):
                    context_lines.append(f"- Seasons: {tv_info['seasons_count']}")
                if tv_info.get('air_years'):
                    context_lines.append(f"- Air Years: {tv_info['air_years']}")

        context = "\n".join(context_lines)

        if len(context_lines) < 3: 
            response = f"I don't have enough information about '{title}' to answer your question properly."
            # Cache the response
            self._query_cache[query] = (response, time.time())
            return response

        # Simplified prompt for faster processing
        meta_prompt = ChatPromptTemplate.from_template("""
        You are a Movie Companion AI. Answer the user's question using the provided context.

        CONTEXT: {context}

        USER QUESTION: "{user_question}"

        INSTRUCTIONS:
        - Answer directly and concisely
        - Use context information to provide accurate details
        - For plot questions, be engaging but avoid major spoilers
        - If context is insufficient, state what information is missing
        - Be conversational and helpful

        ANSWER:
        """)

        try:
            # Use load balancer for response generation if available
            if self.load_balancer:
                # QA response generation is complex - use quality models (LLaMA 4)
                llm = self.load_balancer.get_model("quality")
            else:
                llm = self.groq1
                
            chain = LLMChain(llm=llm, prompt=meta_prompt)
            response = await chain.ainvoke({
                "context": context,
                "user_question": actual_query if actual_query else query,
                "movie_title": title
            })
            
            final_response = response['text'].strip()
            
            # Cache the response
            self._query_cache[query] = (final_response, time.time())
            
            processing_time = time.time() - start_time
            print(f"QA query processed in {processing_time:.2f} seconds with load balanced models")
            
            return final_response

        except Exception as e:
            error_response = f"I encountered an error while trying to answer your question: {str(e)}"
            # Cache error response briefly
            self._query_cache[query] = (error_response, time.time())
            return error_response


# async def main():
#     qa_system = MovieQASystem()



#     qq= ["what is the main story of dune part two",
#             # "In the movie 'The Shawshank Redemption', who is the main character?",
#             # "What happens at the end of Inception?",
#             # "Tell me about the plot of 'The Godfather'",
#             # "Who played the main villain in Dark Knight?",
#             "What's the twist in 'Fight Club'?"]
#     for query in qq:
#         answer = await qa_system.answer_query(query)
#         print(f"Q: {query}")
#         print(f"A: {answer}")
#         await asyncio.sleep(0.5)

# if __name__ == "__main__":
#     asyncio.run(main())