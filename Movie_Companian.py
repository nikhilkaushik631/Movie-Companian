import os
import asyncio
import time
import random
from typing import Dict, Tuple, Optional
from functools import lru_cache
from Multi_Agent import IntegratedQASystem
from Content_Recommend import RAGContentRecommender
from utils import LRUCache, APIRateLimiter
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()

class APILoadBalancer:
    """Enhanced load balancer focused on your existing 3 APIs with aggressive optimization"""
    
    def __init__(self, gemini_api_key1: str, gemini_api_key2: str, groq_api_key: str):
        self.gemini_keys = [gemini_api_key1, gemini_api_key2]
        self.groq_key = groq_api_key
        self.current_gemini_idx = 0
        self.api_usage_count = {"gemini_1": 0, "gemini_2": 0, "groq": 0, "groq_openai": 0, "groq_qwen": 0, "huggingface": 0}
        self.api_errors = {"gemini_1": 0, "gemini_2": 0, "groq": 0, "groq_openai": 0, "groq_qwen": 0, "huggingface": 0}
        
        # Aggressive caching for maximum speed
        self._web_search_cache = {}
        self._web_cache_ttl = 1800  # 30 minutes for web searches
        self._model_response_cache = {}
        self._model_cache_ttl = 600  # 10 minutes for model responses
        
        # Initialize models for different complexity levels
        self._init_models()
    
    def _init_models(self):
        """Initialize models optimized for your existing APIs + Hugging Face"""
        
        # Ultra-fast models for simple tasks (validation, routing)  
        self.fast_models = {
            "groq_fast": ChatGroq(
                model_name="llama-3.1-8b-instant",  # Current fast production model
                api_key=self.groq_key,
                temperature=0.1,
                max_tokens=50  # Increased for better validation
            ),
            "groq_openai_fast": ChatGroq(
                model_name="openai/gpt-oss-20b",  # Fast OpenAI model on Groq
                api_key=self.groq_key,
                temperature=0.1,
                max_tokens=50
            ),
            "gemini_fast_1": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.gemini_keys[0],
                temperature=0.2,
                max_tokens=50  # Very limited for speed
            ),
            "gemini_fast_2": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=self.gemini_keys[1],
                temperature=0.2,
                max_tokens=50  # Very limited for speed
            )
        }
        
        # Add Hugging Face if token is available
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if hf_token:
            try:
                from langchain_huggingface import HuggingFaceEndpoint
                self.fast_models["hf_fast"] = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.2-3B-Instruct",  # Fast lightweight model
                    huggingfacehub_api_token=hf_token,
                    temperature=0.1,
                    max_new_tokens=50
                )
                print("✅ Added Hugging Face API to load balancer")
            except Exception as e:
                print(f"⚠️ Could not add Hugging Face API: {e}")
        
        # Quality models for complex tasks (using current models + HF)
        self.quality_models = {
            "groq_quality": ChatGroq(
                model_name="llama-3.3-70b-versatile",  # Latest high-quality LLaMA model
                api_key=self.groq_key,
                temperature=0.3,
                max_tokens=800  # Increased for better responses
            ),
            "groq_openai_quality": ChatGroq(
                model_name="openai/gpt-oss-120b",  # High-quality OpenAI model on Groq
                api_key=self.groq_key,
                temperature=0.3,
                max_tokens=800
            ),
            "groq_qwen": ChatGroq(
                model_name="qwen/qwen3-32b",  # Qwen model for variety
                api_key=self.groq_key,
                temperature=0.3,
                max_tokens=800
            ),
            "gemini_quality_1": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.gemini_keys[0],
                temperature=0.4,
                max_tokens=800
            ),
            "gemini_quality_2": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.gemini_keys[1], 
                temperature=0.4,
                max_tokens=800
            )
        }
        
        # Add HF quality model if available
        if hf_token:
            try:
                self.quality_models["hf_quality"] = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # Better quality model
                    huggingfacehub_api_token=hf_token,
                    temperature=0.3,
                    max_new_tokens=800
                )
            except Exception as e:
                pass
        
        # Advanced models for very complex tasks (synthesis only)
        self.advanced_models = {
            "groq_advanced": ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                api_key=self.groq_key,
                temperature=0.5,
                max_tokens=1000  # Balanced for quality
            ),
            "gemini_advanced_1": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.gemini_keys[0],
                temperature=0.6,
                max_tokens=1000
            ),
            "gemini_advanced_2": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.gemini_keys[1],
                temperature=0.6,
                max_tokens=1000
            )
        }
        
        # Add HF advanced model if available
        if hf_token:
            try:
                self.advanced_models["hf_advanced"] = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-70B-Instruct",  # Large model for complex tasks
                    huggingfacehub_api_token=hf_token,
                    temperature=0.5,
                    max_new_tokens=1000
                )
            except Exception as e:
                pass
    
    def get_cached_web_search(self, search_query: str):
        """Get cached web search results if available"""
        cache_key = f"web_{hash(search_query)}"
        if cache_key in self._web_search_cache:
            result, timestamp = self._web_search_cache[cache_key]
            if time.time() - timestamp < self._web_cache_ttl:
                return result
        return None
    
    def cache_web_search(self, search_query: str, results):
        """Cache web search results"""
        cache_key = f"web_{hash(search_query)}"
        self._web_search_cache[cache_key] = (results, time.time())
    
    def get_cached_model_response(self, prompt_hash: str):
        """Get cached model response for repeated prompts"""
        if prompt_hash in self._model_response_cache:
            result, timestamp = self._model_response_cache[prompt_hash]
            if time.time() - timestamp < self._model_cache_ttl:
                return result
        return None
    
    def cache_model_response(self, prompt_hash: str, response):
        """Cache model response"""
        self._model_response_cache[prompt_hash] = (response, time.time())

    def get_model(self, complexity: str = "fast", prefer_api: str = None):
        """Get a model based on complexity level with smart load balancing across 4 APIs"""
        
        model_dict = {
            "fast": self.fast_models,
            "quality": self.quality_models, 
            "advanced": self.advanced_models
        }.get(complexity, self.fast_models)
        
        if prefer_api:
            # Try to use preferred API if specified
            for key in model_dict.keys():
                if prefer_api in key:
                    self._update_usage(key)
                    return model_dict[key]
        
        # Smart load balancing for your 4 APIs (Groq + Gemini1 + Gemini2 + HuggingFace)
        if complexity == "fast":
            # For fast tasks, prefer fastest APIs first
            fast_priority = ["groq", "huggingface", "gemini_1", "gemini_2"]
            
            for api in fast_priority:
                if self.api_errors.get(api, 0) < 3:  # If API is healthy
                    model_key = f"{api}_fast"
                    if model_key in model_dict:
                        self._update_usage(api)
                        return model_dict[model_key]
            
            # Fallback to least loaded Gemini
            if self.api_usage_count["gemini_1"] <= self.api_usage_count["gemini_2"]:
                self._update_usage("gemini_1")
                return model_dict["gemini_fast_1"]
            else:
                self._update_usage("gemini_2")
                return model_dict["gemini_fast_2"]
        
        else:
            # For quality/advanced tasks, use least loaded API with quality models
            available_apis = []
            for api in ["groq", "huggingface", "gemini_1", "gemini_2"]:
                model_key = f"{api}_{complexity}"
                if model_key in model_dict and self.api_errors.get(api, 0) < 3:
                    available_apis.append((api, self.api_usage_count.get(api, 0)))
            
            if available_apis:
                # Sort by usage count (ascending) and pick least used
                available_apis.sort(key=lambda x: x[1])
                chosen_api = available_apis[0][0]
                model_key = f"{chosen_api}_{complexity}"
                self._update_usage(chosen_api)
                return model_dict[model_key]
            
            # Final fallback to Groq
            if f"groq_{complexity}" in model_dict:
                self._update_usage("groq")
                return model_dict[f"groq_{complexity}"]
                
            # Ultimate fallback to Gemini
            self._update_usage("gemini_1")
            return model_dict[f"gemini_{complexity}_1"]
    
    def _update_usage(self, api_name: str):
        """Update API usage statistics"""
        if api_name in self.api_usage_count:
            self.api_usage_count[api_name] += 1
    
    def report_error(self, api_name: str):
        """Report API error for fallback logic"""
        if api_name in self.api_errors:
            self.api_errors[api_name] += 1
    
    def get_stats(self):
        """Get enhanced load balancing statistics"""
        total_usage = sum(self.api_usage_count.values())
        return {
            "total_requests": total_usage,
            "api_distribution": self.api_usage_count,
            "error_counts": self.api_errors,
            "web_cache_size": len(self._web_search_cache),
            "model_cache_size": len(self._model_response_cache),
            "load_percentage": {
                api: (count/total_usage*100) if total_usage > 0 else 0 
                for api, count in self.api_usage_count.items()
            }
        }

class MovieChatbot:
    """
    Optimized Movie Chatbot with performance improvements and intelligent load balancing:
    - Smart validation bypass using keyword matching
    - Response caching with TTL
    - Load balanced LLM calls across 2 Gemini + 1 Groq APIs
    - Intelligent model selection based on task complexity
    - Parallel execution where possible
    - Query optimization with keyword-based routing
    """
    
    def __init__(self, 
                google_api_key: str = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key"),
                gemini_api_key: str = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key"), 
                groq_api_key: str = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key"),
                chat_history_limit: int = 15):
            
        self.google_api_key = google_api_key
        self.gemini_api_key = gemini_api_key
        self.groq_api_key = groq_api_key
        self.chat_history_limit = chat_history_limit
        
        # Initialize load balancer with your 3 APIs
        self.load_balancer = APILoadBalancer(
            gemini_api_key1=google_api_key,
            gemini_api_key2=gemini_api_key,
            groq_api_key=groq_api_key
        )

        # Performance optimization: Use LRU caches with size limits
        self._validation_cache = LRUCache(max_size=500, ttl=300)
        self._routing_cache = LRUCache(max_size=500, ttl=300) 
        self._response_cache = LRUCache(max_size=200, ttl=600)  # Longer TTL for responses
        self._cache_ttl = 300  # 5 minutes cache TTL (kept for backward compatibility)
        
        # Performance metrics
        self._performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'bypass_hits': 0,
            'avg_response_time': 0.0,
            'fast_model_usage': 0,
            'quality_model_usage': 0,
            'advanced_model_usage': 0
        }
        
        # Movie keywords for smart validation bypass
        self._movie_keywords = {
            'movie', 'film', 'cinema', 'tv', 'series', 'show', 'episode', 'season',
            'actor', 'actress', 'director', 'cast', 'plot', 'story', 'character',
            'recommend', 'suggestion', 'watch', 'streaming', 'netflix', 'disney',
            'marvel', 'dc', 'imdb', 'rating', 'review', 'genre', 'drama', 'comedy',
            'action', 'thriller', 'horror', 'romance', 'sci-fi', 'fantasy'
        }

        self.recommender = RAGContentRecommender(
            groq_api_key=groq_api_key,
            load_balancer=self.load_balancer
        )
        print("RAGContentRecommender initialized with load balancing.")

        self.movie_system = IntegratedQASystem(
            gemini_api_key=gemini_api_key,
            groq_api_key=groq_api_key,
            load_balancer=self.load_balancer
        )
        print("IntegratedMovieSystem initialized with load balancing.")
        print("All core systems initialized with performance optimizations and load balancing.")
        self.chat_memory = []

        self._initialize_langchain()

    def _initialize_langchain(self):
        """Initialize LangChain components with load balanced models"""

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=self.chat_history_limit
        )

        # Simplified validation prompt for faster processing
        validation_prompt_with_history = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a Movie Query Validator. Determine if a query relates to movies/TV.
                Return ONLY "yes" or "no". Be quick and decisive."""
            ),
            MessagesPlaceholder(variable_name="chat_history"), 
            HumanMessagePromptTemplate.from_template("Query: {query}\nMovie/TV related?")
        ])

        # Use fast models for validation (simple task)
        self.validation_chain = LLMChain(
            llm=self.load_balancer.get_model("fast"),
            prompt=validation_prompt_with_history,
            memory=self.memory, 
            output_key="validation_result"
        )

        # Simplified router prompt
        router_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """Determine query type: "qa" for information/facts, "recommend" for recommendations.
                Return ONLY "qa" or "recommend"."""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        
        # Use fast models for routing (simple task)
        self.router_chain = LLMChain(
            llm=self.load_balancer.get_model("fast"),
            prompt=router_prompt,
            memory=self.memory, 
            output_key="route"
        )

        # Optimized rewriting prompt  
        rewriting_prompt_messages = [
            SystemMessagePromptTemplate.from_template(
                """Add context from conversation history if needed, but PRESERVE the original question format.
                Keep questions as questions, recommendations as recommendations. 
                If query is already clear, return EXACTLY as-is."""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("Query: {query}")
        ]
        rewriting_prompt = ChatPromptTemplate.from_messages(rewriting_prompt_messages)

        # Use quality models for rewriting (medium complexity task)
        self.rewriting_chain = LLMChain(
            llm=self.load_balancer.get_model("quality"),
            prompt=rewriting_prompt,
            memory=self.memory,
            output_key="rewritten_query"
        )
        print("LangChain components initialized with load balanced models.")

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid (legacy method, now handled by LRU cache)"""
        return time.time() - timestamp < self._cache_ttl

    def _clean_expired_cache(self):
        """Clean expired cache entries (now handled automatically by LRU cache)"""
        # LRU cache handles expiration automatically, but we can force cleanup
        pass

    def _smart_validation_bypass(self, query: str) -> Optional[bool]:
        """Fast keyword-based validation bypass for obvious movie queries"""
        query_lower = query.lower()
        
        # If contains movie keywords, likely valid
        if any(keyword in query_lower for keyword in self._movie_keywords):
            self._performance_stats['bypass_hits'] += 1
            return True
            
        # If asking about non-entertainment topics, likely invalid
        non_movie_indicators = ['weather', 'politics', 'math', 'science', 'cooking', 'recipe']
        if any(indicator in query_lower for indicator in non_movie_indicators):
            self._performance_stats['bypass_hits'] += 1
            return False
            
        return None  # Need LLM validation

    async def validate_movie_query(self, query: str) -> bool:
        """Optimized validation with caching and smart bypass"""
        # Check cache first
        cached_result = self._validation_cache.get(query)
        if cached_result is not None:
            return cached_result
        
        # Try smart bypass
        bypass_result = self._smart_validation_bypass(query)
        if bypass_result is not None:
            self._validation_cache.set(query, bypass_result)
            return bypass_result
        
        # Fallback to LLM validation with load balancing
        try:
            self._performance_stats['fast_model_usage'] += 1
            # Get a fresh fast model for validation
            validation_model = self.load_balancer.get_model("fast")
            validation_chain = LLMChain(
                llm=validation_model,
                prompt=self.validation_chain.prompt,
                memory=self.memory,
                output_key="validation_result"
            )
            
            result = await validation_chain.ainvoke({"query": query})
            validation_result = result.get("validation_result", "").strip().lower()
            positive_indicators = ["yes", "y", "true", "valid", "related"]
            is_valid = any(indicator in validation_result for indicator in positive_indicators)
            
            # Cache result
            self._validation_cache.set(query, is_valid)
            return is_valid
        except Exception as e:
            print(f"Error in validation: {e}")
            # Report error to load balancer
            self.load_balancer.report_error("fast_model")
            return True  # Default to valid for better UX

    async def rewrite_query(self, query: str) -> str:
        """Optimized query rewriting with load balanced quality models"""
        try:
            current_history = self.memory.load_memory_variables({}).get("chat_history", [])
            if not current_history:
                return query 

            self._performance_stats['quality_model_usage'] += 1
            # Get a fresh quality model for rewriting
            rewriting_model = self.load_balancer.get_model("quality")
            rewriting_chain = LLMChain(
                llm=rewriting_model,
                prompt=self.rewriting_chain.prompt,
                memory=self.memory,
                output_key="rewritten_query"
            )
            
            result = await rewriting_chain.ainvoke({"query": query})
            rewritten_query = result.get("rewritten_query", "").strip()
            return rewritten_query if rewritten_query else query
        except Exception as e:
            print(f"Error rewriting query: {e}")
            self.load_balancer.report_error("quality_model")
            return query 

    def _smart_routing_bypass(self, query: str) -> Optional[str]:
        """Fast keyword-based routing for obvious cases"""
        query_lower = query.lower()
        
        # Recommendation indicators
        recommend_keywords = ['recommend', 'suggest', 'similar', 'like', 'best', 'top', 'good']
        if any(keyword in query_lower for keyword in recommend_keywords):
            self._performance_stats['bypass_hits'] += 1
            return "recommend"
        
        # QA indicators
        qa_keywords = ['who', 'what', 'when', 'where', 'how', 'plot', 'story', 'about', 'tell me']
        if any(keyword in query_lower for keyword in qa_keywords):
            self._performance_stats['bypass_hits'] += 1
            return "qa"
            
        return None  # Need LLM routing

    async def route_query(self, query: str) -> str:
        """Optimized query routing with caching and smart bypass"""
        # Check cache first
        cached_result = self._routing_cache.get(query)
        if cached_result is not None:
            return cached_result
        
        # Try smart bypass
        bypass_result = self._smart_routing_bypass(query)
        if bypass_result is not None:
            self._routing_cache.set(query, bypass_result)
            return bypass_result
        
        # Fallback to LLM routing with load balancing
        try:
            self._performance_stats['fast_model_usage'] += 1
            # Get a fresh fast model for routing
            routing_model = self.load_balancer.get_model("fast")
            routing_chain = LLMChain(
                llm=routing_model,
                prompt=self.router_chain.prompt,
                memory=self.memory,
                output_key="route"
            )
            
            result = await routing_chain.ainvoke({"query": query})
            route = result.get("route", "").strip().lower()
            route = route if route in ["qa", "recommend"] else "qa"
            
            # Cache result
            self._routing_cache.set(query, route)
            return route
        except Exception as e:
            print(f"Error routing query: {e}")
            self.load_balancer.report_error("fast_model")
            return "qa"

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics including load balancing"""
        stats = self._performance_stats.copy()
        if stats['total_queries'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
            stats['bypass_hit_rate'] = stats['bypass_hits'] / stats['total_queries']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['bypass_hit_rate'] = 0.0
        
        # Add load balancing stats
        stats['load_balancer'] = self.load_balancer.get_stats()
        return stats

    def reset_performance_stats(self):
        """Reset performance statistics"""
        self._performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'bypass_hits': 0,
            'avg_response_time': 0.0,
            'fast_model_usage': 0,
            'quality_model_usage': 0,
            'advanced_model_usage': 0
        }

    async def chat(self, user_input: str, session_id: str = None) -> str:
        """Optimized chat processing with load balanced models and performance improvements"""
        
        # Update performance stats
        self._performance_stats['total_queries'] += 1
        
        # Handle special commands quickly
        if user_input.lower() == 'clear':
            self.memory.clear()
            # Clear caches too
            self._validation_cache.clear()
            self._routing_cache.clear()
            self._response_cache.clear()
            return "Our conversation has been cleared. What would you like to know about movies or TV shows?"

        if user_input.lower() in ['q', 'quit', 'exit']:
            return "Goodbye! Enjoy your movie watching!"
        
        # Show performance stats including load balancing on request
        if user_input.lower() in ['stats', 'performance']:
            stats = self.get_performance_stats()
            load_stats = stats['load_balancer']
            api_dist = load_stats.get('api_distribution', {})
            
            # Create a clean distribution display
            dist_text = " | ".join([
                f"{api.title()}: {count}" for api, count in api_dist.items() if count > 0
            ])
            
            return f"""🚀 Performance Stats:
• Queries: {stats['total_queries']} | Cache hits: {stats.get('cache_hit_rate', 0):.1%} | Bypasses: {stats.get('bypass_hit_rate', 0):.1%}
• Avg response: {stats.get('avg_response_time', 0):.2f}s
• Model usage: Fast({stats.get('fast_model_usage', 0)}) Quality({stats.get('quality_model_usage', 0)}) Advanced({stats.get('advanced_model_usage', 0)})
• API distribution: {dist_text}
• Total API calls: {load_stats.get('total_requests', 0)} | Caches: Web({load_stats.get('web_cache_size', 0)}) Model({load_stats.get('model_cache_size', 0)})
• 4 APIs active: Groq (LLaMA 4) + 2× Gemini + Hugging Face"""

        # Clean expired cache entries periodically
        self._clean_expired_cache()
        
        # Check response cache first
        cache_key = f"{user_input}_{len(self.memory.load_memory_variables({}).get('chat_history', []))}"
        cached_response = self._response_cache.get(cache_key)
        if cached_response is not None:
            self._performance_stats['cache_hits'] += 1
            return cached_response

        start_time = time.time()
        recommendation_cards = []  # Initialize recommendation cards list
        
        # Parallel execution for independent operations where possible
        is_movie_related = await self.validate_movie_query(user_input)

        if not is_movie_related:
            response_text = "I'm specialized in movies and TV shows. Could you ask me something related to films, series, actors, directors, or similar topics?"
            self.memory.save_context(
                {"input": user_input},
                {"output": response_text}
            )
            return {
                "response": response_text,
                "session_id": session_id,
                "query_type": "invalid",
                "is_movie_related": False,
                "recommendation_cards": []
            }

        # Route BEFORE rewriting to preserve question structure
        query_type = await self.route_query(user_input)
        # Only rewrite if we have conversation context
        resolved_query = await self.rewrite_query(user_input) if len(self.memory.load_memory_variables({}).get('chat_history', [])) > 0 else user_input

        response_text = ""
        try:
            if query_type == "qa":
                # QA uses advanced models for better quality
                self._performance_stats['advanced_model_usage'] += 1
                if hasattr(self.movie_system, 'aprocess_query'):
                    result = await self.movie_system.aprocess_query(resolved_query)
                else:
                    result = await self.movie_system.process_query(resolved_query)

                response_text = result.get('response', 'I could not find information about that.')
            else:  
                # Recommendations use quality models
                self._performance_stats['quality_model_usage'] += 1
                if hasattr(self.recommender, 'recommend_with_metadata'):
                    # Use new method that returns both text and card data
                    recommendation_data = await self.recommender.recommend_with_metadata(resolved_query, top_k=5)
                    response_text = recommendation_data.get('response', 'No recommendations found.')
                    recommendation_cards = recommendation_data.get('recommendations', [])
                elif hasattr(self.recommender, 'arecommend'):
                    response_text = await self.recommender.arecommend(resolved_query, top_k=5)
                    recommendation_cards = []
                else:
                    response_text = await self.recommender.recommend(resolved_query, 5)
                    recommendation_cards = []
                    
        except Exception as e:
            print(f"Error during QA or Recommendation processing: {e}")
            response_text = "Sorry, I encountered an issue while processing your request."
            recommendation_cards = []

        # Cache the response
        self._response_cache.set(cache_key, response_text)
        
        # Save to memory
        self.memory.save_context({"input": user_input}, {"output": response_text})
        
        # Update performance metrics
        processing_time = time.time() - start_time
        total_queries = self._performance_stats['total_queries']
        current_avg = self._performance_stats['avg_response_time']
        self._performance_stats['avg_response_time'] = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        
        print(f"Query processed in {processing_time:.2f} seconds using load balanced models")

        # Return enhanced response with recommendation cards
        return {
            "response": response_text,
            "session_id": session_id,
            "query_type": query_type,
            "is_movie_related": is_movie_related,
            "recommendation_cards": recommendation_cards if query_type == "recommend" else []
        }


    async def start_chatbot_cli(self):
        """Start an interactive chatbot CLI session"""
        print("MovieBot")
        print("------------------------------------------------")
        print("Ask me anything about movies and TV shows, or ask for recommendations!")
        print("Type 'q' to quit or 'clear' to reset our conversation.")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nYou: ")     
            except KeyboardInterrupt:
                print("\nMovieBot: Exiting...")
                break

            if user_input.lower() == 'q':
                print("\nMovieBot: Goodbye! Enjoy your movie watching!")
                break

            response = await self.chat(user_input)
            print(f"\nMovieBot: {response}")



async def main_async():
    """Asynchronous main function to run the chatbot."""
    chatbot = MovieChatbot()

#     queries_to_test = [
#         "Who directed Inception?",
#         "Did he direct any other sci-fi movies?",
#         "Recommend me movies like those",
#         "What is the capital of France?",
#         "What is the capital of Italy?",
#         "movies with great cgi like dune part two",
#         "who directed dune part two?",
#         "japanese anime like bleach",
#         "movies like interstellar",
#         "In the movie 'The Shawshank Redemption', who is the main character?",
#         "What happens at the end of Inception?",
#         "Tell me about the plot of 'The Godfather'",
#     ]

#     for query in queries_to_test:
#         print(f"\nUser: {query}") 
#         response = await chatbot.chat(query)
#         print(f"MovieBot: {response}") 
#         await asyncio.sleep(2)

    await chatbot.start_chatbot_cli()

if __name__ == "__main__":
    # To run the async version:
    try:
        #  import nest_asyncio
        #  nest_asyncio.apply()
        #  asyncio.run(main_async())
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Exited by user.")