import os
import warnings
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
from QA import MovieQASystem
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import asyncio
import aiohttp
import httpx
from dotenv import load_dotenv
import time
import hashlib

# Modern search alternatives
try:
    from ddgs import DDGS  # New package name
    USE_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # Fallback to old package
        USE_DDGS = True
    except ImportError:
        USE_DDGS = False

load_dotenv()
warnings.filterwarnings("ignore")

class MovieQueryState(BaseModel):
    """State for the movie query processing system"""
    original_query: str = Field(description="Original user query")
    db_response: str = Field(default="", description="Response from the movie database")
    needs_web_search: bool = Field(default=False, description="Whether web search is needed")
    search_query: str = Field(default="", description="Optimized search query for web search")
    raw_search_results: Dict[str, Any] = Field(default_factory=dict, description="Raw search results from web")
    cleaned_content: List[str] = Field(default_factory=list, description="Cleaned relevant paragraphs")
    final_response: str = Field(default="", description="Final synthesized response")
    sources: List[Dict] = Field(default_factory=list, description="Sources used")
    chat_history: List[str] = Field(default_factory=list, description="Chat conversation history")
    analysis_result: Dict = Field(default_factory=dict, description="Query analysis results")

class IntegratedQASystem:
    """Ultra-optimized Movie information system with lightning-fast web search"""
    
    def __init__(
        self,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key"),
        groq_api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key"),
        chat_history_limit: int = 10,
        load_balancer=None  # Accept load balancer from main class
    ):
        self.qa_system = MovieQASystem(
            google_api_key=gemini_api_key,
            groq_api_key=groq_api_key,
            load_balancer=load_balancer
        )
        self.chat_history_limit = chat_history_limit
        self.load_balancer = load_balancer
        
        # Async HTTP client for web searches
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={"User-Agent": "MovieQA/1.0"}
        )

        # Initialize models - use load balancer if available, otherwise fallback
        if self.load_balancer:
            print("Multi-Agent System: Using load balanced models with web optimization")
        else:
            # Fallback to direct model initialization
            self.groq1 = ChatGroq(
                model_name="llama3-8b-8192",  # Faster model
                api_key=groq_api_key,
                temperature=0.3,
                max_tokens=300  # Further reduced for speed
            )

            self.groq2 = ChatGroq(
                model_name="llama3-8b-8192",  # Consistent fast model
                api_key=groq_api_key,
                temperature=0.5,
                max_tokens=500  # Reduced for content extraction
            )
            
            self.gemini = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_api_key,
                temperature=0.7
            )
            print("Multi-Agent System: Using fallback direct models")

        # Performance optimization: Smart decision keywords
        self._sufficient_keywords = {
            'plot', 'story', 'cast', 'actor', 'actress', 'director', 'character',
            'release', 'year', 'genre', 'runtime', 'rating', 'based on'
        }
        
        self._needs_web_keywords = {
            'latest', 'recent', 'new', 'current', 'streaming', 'where to watch',
            'best', 'top', 'ranking', 'review', 'news', 'upcoming', 'now', 'today'
        }

        # Enhanced web search optimization
        self._web_search_cache = {}
        self._web_cache_ttl = 1800  # 30 minutes
        
        # Pre-defined movie APIs for faster searches
        self._movie_apis = [
            "https://www.imdb.com/find/?q=",
            "https://www.themoviedb.org/search?query=",
            "https://www.rottentomatoes.com/search/?search="
        ]

        self.agent = self._build_graph()
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for web search"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _get_cached_search(self, query: str):
        """Get cached web search results"""
        cache_key = self._get_cache_key(query)
        if cache_key in self._web_search_cache:
            result, timestamp = self._web_search_cache[cache_key]
            if time.time() - timestamp < self._web_cache_ttl:
                print(f"🚀 Using cached web search for: {query}")
                return result
        return None
    
    def _cache_search(self, query: str, results):
        """Cache web search results"""
        cache_key = self._get_cache_key(query)
        self._web_search_cache[cache_key] = (results, time.time())

    def _smart_analysis_bypass(self, query: str, db_response: str) -> Dict:
        """Ultra-fast keyword-based analysis to bypass LLM when obvious"""
        query_lower = query.lower()
        
        # Check if DB response indicates missing info
        insufficient_indicators = [
            "couldn't find", "don't have", "not available", "unknown",
            "please check", "error", "insufficient", "no information"
        ]
        
        if any(indicator in db_response.lower() for indicator in insufficient_indicators):
            return {
                "needs_web_search": True,
                "reason": "Database lacks sufficient information",
                "search_query": f"{query} movie tv information"
            }
        
        # Check for obvious web search needs (current/trending info)
        if any(keyword in query_lower for keyword in self._needs_web_keywords):
            return {
                "needs_web_search": True,
                "reason": "Query requires current/trending information",
                "search_query": f"{query} movie tv latest"
            }
        
        # Check for likely sufficient database info
        plot_keywords = ['plot', 'story', 'ending', 'happens', 'character', 'main character']
        if (any(keyword in query_lower for keyword in plot_keywords) and 
            len(db_response) > 50):  # More aggressive for plot questions
            return {
                "needs_web_search": False,
                "reason": "Plot information should be in database",
                "search_query": ""
            }
            
        if (any(keyword in query_lower for keyword in self._sufficient_keywords) and 
            len(db_response) > 100 and not any(indicator in db_response.lower() for indicator in insufficient_indicators)):
            return {
                "needs_web_search": False,
                "reason": "Database provides sufficient information",
                "search_query": ""
            }
        
        return None  # Need LLM analysis

    async def _query_database_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for querying the existing movie database"""
        try:
            db_response = await self.qa_system.answer_query(state.original_query)
            state.db_response = db_response
        except Exception as e:
            state.db_response = f"Error querying database: {str(e)}"
        
        return state

    async def _analyze_query_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Ultra-optimized agent for analyzing if web search is needed"""
        
        # Try smart bypass first (90% of cases should be handled here)
        bypass_result = self._smart_analysis_bypass(state.original_query, state.db_response)
        if bypass_result:
            state.needs_web_search = bypass_result["needs_web_search"]
            state.search_query = bypass_result["search_query"]
            state.analysis_result = bypass_result
            if state.needs_web_search:
                print("🔍 Searching web for latest information...")
            return state

        # Super simplified LLM analysis prompt for remaining cases
        analysis_prompt = ChatPromptTemplate.from_template("""
        Query: {original_query}
        DB: {db_response}

        Need web search? Response format: YES or NO
        If YES, add search terms after a comma.
        """)
        
        try:
            # Use load balancer for analysis if available (fast models only)
            if self.load_balancer:
                llm = self.load_balancer.get_model("fast")  # Use fastest models
            else:
                llm = self.groq1
                
            chain = LLMChain(llm=llm, prompt=analysis_prompt)
            response = await chain.ainvoke({
                "original_query": state.original_query,
                "db_response": state.db_response[:500]  # Truncate for speed
            })
            
            response_text = response['text'].strip().upper()
            
            if "YES" in response_text:
                state.needs_web_search = True
                # Extract search terms after comma
                if "," in response_text:
                    search_terms = response_text.split(",", 1)[1].strip()
                    state.search_query = search_terms if search_terms else f"{state.original_query} movie"
                else:
                    state.search_query = f"{state.original_query} movie"
                print("🔍 Searching web for latest information...")
            else:
                state.needs_web_search = False
                state.search_query = ""
                
        except Exception as e:
            print(f"Analysis error, defaulting to web search: {e}")
            state.needs_web_search = True
            state.search_query = f"{state.original_query} movie"
        
        return state

    def _route_based_on_analysis(self, state: MovieQueryState) -> str:
        """Determine the next node based on analysis results"""
        if state.needs_web_search:
            return "needs_web_search"
        else:
            return "sufficient_info"

    async def _lightning_web_search(self, search_query: str) -> Dict[str, Any]:
        """Lightning-fast web search with aggressive optimization"""
        
        # Strategy 1: Check cache first (30 min cache)
        cached_result = self._get_cached_search(search_query)
        if cached_result:
            return cached_result
            
        search_results = {}
        
        # Strategy 2: Ultra-fast search with 2-second timeout
        try:
            # Fast DuckDuckGo search with aggressive timeout
            search_task = asyncio.create_task(self._ddg_search(search_query))
            
            # Wait maximum 2 seconds for search
            results = await asyncio.wait_for(search_task, timeout=2.0)
            if results:
                search_results.update(results)
                
        except asyncio.TimeoutError:
            print("⚡ Web search timeout (2s) - using fallback response")
            # Provide a quick fallback response
            search_results["fallback"] = {
                "url": f"https://www.google.com/search?q={search_query.replace(' ', '+')}",
                "title": f"Search for {search_query}",
                "content": f"For the latest information about {search_query}, please check recent sources online.",
                "source": "Fallback"
            }
        except Exception as e:
            print(f"⚡ Search error: {e}")
            
        # Cache results even if minimal
        self._cache_search(search_query, search_results)
        return search_results
    
    async def _ddg_search(self, query: str) -> Dict[str, Any]:
        """Ultra-fast DuckDuckGo search"""
        try:
            # Use the new ddgs package for faster searches
            with DDGS() as ddgs:
                # Get only 1 result for maximum speed
                results = list(ddgs.text(query, max_results=1))
                
            search_results = {}
            for i, result in enumerate(results):
                search_results[f"web_{i}"] = {
                    "url": result.get("href", ""),
                    "title": result.get("title", ""),
                    "content": result.get("body", "")[:800],  # Aggressive truncation
                    "source": "DuckDuckGo"
                }
            return search_results
            
        except Exception as e:
            return {}

    async def _web_search_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Ultra-optimized agent for lightning-fast web searches"""
        
        print(f"⚡ Lightning search: {state.search_query}")
        start_time = time.time()
        
        # Use lightning search with 2-second max
        search_results = await self._lightning_web_search(state.search_query)
        
        search_time = time.time() - start_time
        print(f"🚀 Web search completed in {search_time:.2f}s")
        
        state.raw_search_results = search_results
        return state

    async def _content_extraction_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Ultra-optimized agent for extracting relevant content"""
        if not state.raw_search_results:
            return state
        
        # Super simplified extraction - just use the content directly
        extracted_content = []
        sources = []
        
        for key, data in list(state.raw_search_results.items())[:2]:  # Max 2 sources
            content = data.get('content', '')
            if content and len(content) > 50:  # Only meaningful content
                # Extract key sentences
                sentences = content.split('. ')[:3]  # First 3 sentences only
                extracted_content.extend(sentences)
                
                sources.append({
                    "url": data.get('url', ''),
                    "title": data.get('title', 'Unknown'),
                    "source": data.get('source', 'Web')
                })
        
        state.cleaned_content = extracted_content[:5]  # Max 5 lines
        state.sources = sources
        
        return state

    async def _response_synthesis_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Ultra-optimized agent for synthesizing the final response"""
        
        # Super simplified synthesis prompt
        response_synthesis_prompt = ChatPromptTemplate.from_template("""
        Question: {original_query}
        
        Database: {db_response}
        
        Web info: {web_search_info}
        
        Answer briefly and directly:
        """)
        
        web_search_info = ""
        if state.cleaned_content:
            web_search_info = " ".join(state.cleaned_content[:3])  # Only first 3 items
        
        try:
            # Use load balancer for synthesis if available
            if self.load_balancer:
                # Use quality models but with reduced tokens for speed
                llm = self.load_balancer.get_model("quality")
            else:
                llm = self.gemini
                
            chain = LLMChain(llm=llm, prompt=response_synthesis_prompt)
            response = await chain.ainvoke({
                "original_query": state.original_query,
                "db_response": state.db_response[:800],  # Truncate for speed
                "web_search_info": web_search_info[:500]  # Truncate for speed
            })
            
            state.final_response = response['text'].strip()
        except Exception as e:
            print(f"Response synthesis error: {e}")
            state.final_response = state.db_response or "I couldn't find sufficient information."
        
        return state

    def _build_graph(self):
        """Build the ultra-optimized LangGraph agent workflow"""
        workflow = StateGraph(MovieQueryState)
        
        # Add nodes
        workflow.add_node("query_database", self._query_database_agent)
        workflow.add_node("analyze_query", self._analyze_query_agent)
        workflow.add_node("web_search", self._web_search_agent)
        workflow.add_node("content_extraction", self._content_extraction_agent)
        workflow.add_node("response_synthesis", self._response_synthesis_agent)
        
        # Define edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_based_on_analysis,
            {
                "needs_web_search": "web_search",
                "sufficient_info": "response_synthesis"
            }
        )
        
        workflow.add_edge("query_database", "analyze_query")
        workflow.add_edge("web_search", "content_extraction")
        workflow.add_edge("content_extraction", "response_synthesis")
        workflow.add_edge("response_synthesis", END)
        
        workflow.set_entry_point("query_database")
        
        return workflow.compile()

    async def process_query(self, query: str, chat_history: List[str] = None) -> Dict[str, Any]:
        """Process a user query through the ultra-optimized integrated system"""
        if chat_history is None:
            chat_history = []

        initial_state = MovieQueryState(
            original_query=query,
            chat_history=chat_history[-self.chat_history_limit:] if chat_history else []  # Limit history
        )

        try:
            start_time = time.time()
            result = await self.agent.ainvoke(initial_state)
            total_time = time.time() - start_time
            
            print(f"🚀 Total multi-agent processing: {total_time:.2f}s")
            
            # Update chat history efficiently
            chat_history.append(f"User: {query}")
            chat_history.append(f"Bot: {result['final_response']}")
            
            # Limit chat history size
            if len(chat_history) > self.chat_history_limit * 2:  
                chat_history = chat_history[-(self.chat_history_limit * 2):]
            
            return {
                "response": result["final_response"],
                "sources": result.get("sources", []),
                "used_web_search": result.get("analysis_result", {}).get("needs_web_search", False),
                "processing_time": total_time
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                "response": error_msg,
                "sources": []
            }

    # Add alias for async processing
    async def aprocess_query(self, query: str, chat_history: List[str] = None) -> Dict[str, Any]:
        """Alias for async process_query method."""
        return await self.process_query(query, chat_history)


# async def main():
#     movie_system = IntegratedQASystem()

#     print("Movie Information System")
#     print("-----------------------")
#     chat_memory = []
#     try:
#         while True:
#             query = input("\nEnter your movie question (or 'q' to quit, 'clear' to reset conversation): ")
            
#             if query.lower() == 'q':
#                 break
                
#             if query.lower() == 'clear':
#                 chat_memory = []
#                 print("Chat history cleared.")
#                 continue
            
#             print("\nProcessing your query...")
#             result = await movie_system.process_query(query, chat_history=chat_memory)
            
#             print("\nRESPONSE:")
#             print(f"A: {result['response']}")
            
#             if result['sources']:
#                 print("\nSources:")
#                 for source in result['sources']:
#                     print(f"- {source.get('title', 'Unknown')}: {source.get('url', 'No URL')}")
            
#     except Exception as e:
#         print(f"IntegratedQASystem error: {e}")


# if __name__ == "__main__":
#     asyncio.run(main())