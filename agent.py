import os
import warnings
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import json
from QNA import MovieQASystem
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore")

class MovieQueryState(BaseModel):
    """State for the movie query processing system"""
    original_query: str = Field(description="Original user query")
    db_response: str = Field(default="", description="Response from the movie database")
    needs_web_search: bool = Field(default=False, description="Whether web search is needed")
    search_query: str = Field(default="", description="Optimized search query for web search")
    raw_search_results: Dict[str, str] = Field(default_factory=dict, description="Raw search results from web")
    cleaned_content: List[str] = Field(default_factory=list, description="Cleaned relevant paragraphs")
    final_response: str = Field(default="", description="Final synthesized response")
    sources: List[Dict] = Field(default_factory=list, description="Sources used")
    chat_history: List[str] = Field(default_factory=list, description="Chat conversation history")
    analysis_result: Dict = Field(default_factory=dict, description="Query analysis results")

class IntegratedMovieSystem:
    """Movie information system using LangGraph for agent orchestration"""
    
    def __init__(
        self,
        movie_qa_system,
        gemini_api_key: str,
        groq_api_key: str,
        chat_history_limit: int = 5
    ):
        self.qa_system = movie_qa_system
        self.chat_history_limit = chat_history_limit

        self.groq1 = ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key
        )

        self.groq2 = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )
        
        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
        )

        self.agent = self._build_graph()
    
    def _query_database_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for querying the existing movie database"""
        try:
            db_response = self.qa_system.answer_query(state.original_query)
            state.db_response = db_response
        except Exception as e:
            state.db_response = f"Error querying database: {str(e)}"
        
        return state
    
    def _analyze_query_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for analyzing if web search is needed"""

        analysis_prompt = ChatPromptTemplate.from_template("""
        You are a query analyzer agent. You work with queries related to Movies and Tv Series. Analyze this user query and the database response to determine if web search is needed.

        User Query: {original_query}

        Database Response: {db_response}

        Your analysis tasks:
        1. EVALUATION: Does the database response completely answer the specific user query? Consider whether it:
          - Provides the specific information requested (names, dates, lists, etc.)
          - Includes all requested details (not just general descriptions)
          - Contains accurate and complete information
          - Addresses the actual intent of the query (not just the topic)

        2. SPECIFICITY CHECK: Is the user asking for specific information like:
          - Lists of items, people, or entities (e.g., "list all characters", "who are the members")
          - Detailed facts (e.g., "when did", "where was", "how many")
          - Comparisons or relationships (e.g., "difference between", "connection between")
          - Chronological information (e.g., "timeline", "order of events")

        3. MISSING INFORMATION: Identify any specific information requested but not provided.

        Return a valid JSON with these fields ONLY:
        - "needs_web_search": true/false (set to true if the database response doesn't fully answer the specific query)
        - "reason": explain exactly what information is missing or incomplete
        - "entities": list of key entities to search for
        - "search_query": an optimized search query focused on the SPECIFIC missing information
        """)
        
        chain = LLMChain(llm=self.groq2, prompt=analysis_prompt)
        response = chain.invoke({
            "original_query": state.original_query,
            "db_response": state.db_response
        })
        
        try:
            # Parse the JSON response
            result = json.loads(response['text'])
            state.needs_web_search = result.get("needs_web_search", False)
            state.search_query = result.get("search_query", state.original_query + " movie info")
            state.analysis_result = result
        except Exception as e:
            # Fallback if JSON parsing fails
            state.needs_web_search = "insufficient" in response['text'].lower() or "outdated" in response['text'].lower()
            state.search_query = state.original_query + " movie info"
            state.analysis_result = {
                "needs_web_search": state.needs_web_search,
                "entities": [e.strip() for e in state.original_query.split() if len(e) > 3],
                "search_query": state.search_query,
                "reason": "Analysis fallback due to parsing error"
            }
        
        return state
    

    def _route_based_on_analysis(self, state: MovieQueryState) -> str:
        """Determine the next node based on analysis results"""
        if state.needs_web_search:
            return "needs_web_search"
        else:
            return "sufficient_info"
    

    def _web_search_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for performing web searches using DuckDuckGo"""
        search_results = {}
        
        try:
            print(f"Searching DuckDuckGo for: {state.search_query}")
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(
                    state.search_query, 
                    region='wt-wt', 
                    safesearch='Moderate', 
                    max_results=3
                )]
            
            urls = [r["href"] for r in results]

            for url in urls:
                try:
                    print(f"Scraping: {url}")
                    response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(response.text, "html.parser")
                    content = soup.get_text(separator="\n", strip=True)
                    
                    # Limit content size
                    if len(content) > 10000:
                        content = content[:10000] + "... [content truncated]"
                    
                    result = {
                        "url": url,
                        "title": soup.title.string if soup.title else "Unknown",
                        "content": content,
                        "query": state.search_query,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    search_results[url] = result
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
        except Exception as e:
            print(f"Web search error: {e}")
        
        state.raw_search_results = search_results
        return state
    

    def _content_extraction_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for extracting relevant paragraphs from search results"""
        if not state.raw_search_results:
            return state
        
        # Format the raw search results for processing
        formatted_results = ""
        for url, data in state.raw_search_results.items():
            formatted_results += f"Source: {url}\nTitle: {data.get('title', 'Unknown')}\n"
            formatted_results += f"Content: {data.get('content', '')}...\n\n"
        
        content_extraction_prompt = ChatPromptTemplate.from_template("""
            You are a Data Extraction Specialist for movies and TV shows. Your job is to extract SPECIFIC information from search results.

            Original Query: {original_query}
            Search Query Used: {search_query}
            Reason for search: {reason}

            Search results:
            {raw_search_results}

            INSTRUCTIONS:
            1. Focus on extracting PRECISELY the information requested in the query
            2. If the query asks for a list of characters, extract the complete list
            3. If the query asks for specific details, extract those exact details
            4. If the query is about characters, cast, crew, or any group, identify ALL members
            5. Organize information clearly with proper names, descriptions, and relevant details
            6. Prioritize complete and accurate information over general descriptions
            7. Extract from all sources and consolidate information to be comprehensive

            EXTRACTED INFORMATION (be specific and complete):
            """)

        
        chain = LLMChain(llm=self.groq1, prompt=content_extraction_prompt)
        response = chain.invoke({
            "original_query": state.original_query,
            "search_query": state.search_query,
            "reason": state.analysis_result.get("reason", "Information gathering"),
            "raw_search_results": formatted_results
        })
        
        lines = response['text'].strip().splitlines()
        state.cleaned_content = [line.strip() for line in lines if line.strip()]

        sources = []
        for url, data in state.raw_search_results.items():
            sources.append({
                "url": url,
                "title": data.get("title", "Unknown"),
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            })
        
        state.sources = sources
        
        return state
    

    def _response_synthesis_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for synthesizing the final response"""
        response_synthesis_prompt = ChatPromptTemplate.from_template("""
        You are a Movie Information Specialist. Your job is to synthesize a helpful, accurate response.

        Original Query: {original_query}
        
        Information from Database:
        {db_response}
        
        {web_search_info}
        
        Chat History:
        {chat_history}

        Guidelines:
        - For single-fact questions: Answer consisely wtih that fact.
        - For factual questions: Be direct and precise.
        - For plot questions: Narrate like a storyteller without major spoilers.
        - For info based questions: Provide detailed information.
        - For opinion questions: Present balanced views.
        - For analytical questions: Provide deep insights.
        - When combining database and web information, prioritize recency and accuracy.
        - Response should be in a paragraph form, not bullet points.
        - Strike a balance between comprehensive information and readability.
        
        Final Response:
        """)
        
        # Format web search info if available
        web_search_info = ""
        if state.cleaned_content:
            web_search_info = "Information from Web Search:\n" + "\n".join(state.cleaned_content)
        
        chain = LLMChain(llm=self.groq2, prompt=response_synthesis_prompt)
        response = chain.invoke({
            "original_query": state.original_query,
            "db_response": state.db_response,
            "web_search_info": web_search_info,
            "chat_history": "\n".join(state.chat_history) if state.chat_history else "None"
        })
        
        state.final_response = response['text'].strip()
        return state
    


    def _build_graph(self):
        """Build the LangGraph agent workflow"""
        workflow = StateGraph(MovieQueryState)
        
        # Add nodes to the graph
        workflow.add_node("query_database", self._query_database_agent)
        workflow.add_node("analyze_query", self._analyze_query_agent)
        workflow.add_node("web_search", self._web_search_agent)
        workflow.add_node("content_extraction", self._content_extraction_agent)
        workflow.add_node("response_synthesis", self._response_synthesis_agent)
        
        # Define the edges in the graph (conditional routing)
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


    def rewrite_followup_query(self, query: str, chat_history: List[str]) -> str:
        """Rewrites a followup query by incorporating context from the chat history"""
        if not chat_history:
            return query
            
        rewriting_prompt = ChatPromptTemplate.from_template("""
            You are an expert query rewriter for a movie and TV information system. Given the conversation history and the new followup query, 
            reformulate the query so that it is self-contained and includes all necessary context to be answered properly.

            CONVERSATION HISTORY:
            {chat_history}

            CURRENT FOLLOW-UP QUERY:
            {query}

            INSTRUCTIONS:
            1. Identify all implicit references in the query (pronouns, implied subjects, contextual terms)
            2. Determine the specific information being requested (facts, lists, comparisons, explanations)
            3. Incorporate ALL relevant context from the conversation history
            4. For requests seeking detailed information, explicitly include that in the rewrite
            5. Ensure the rewritten query can stand alone and conveys the full intent

            REWRITTEN QUERY (make it detailed, explicit, and comprehensive):
                """)
        
        chain = LLMChain(llm=self.gemini, prompt=rewriting_prompt)
        response = chain.invoke({
            "chat_history": "\n".join(chat_history),
            "query": query
        })
        
        return response["text"].strip()
    
    def validate_query(self, query: str, chat_history: List[str] = None) -> bool:
        """
        Uses an LLM to check if the query is related to movies or TV series.
        For follow-up queries, analyzes the chat history for context.
        """

        if chat_history:
            recent_exchanges = "\n".join(chat_history[-min(4, len(chat_history)):])
            validation_prompt = ChatPromptTemplate.from_template("""
            You are a Movie Query Validator. Determine if the following query is related to movies or TV series.
            
            Recent conversation:
            {chat_history}
            
            Current query: {query}
            
            This could be a follow-up question to the previous movie conversation.
            Consider both the query itself and its context in the conversation.
            
            Return "yes" if it's related to movies/TV (either directly or as a follow-up), or "no" if it's not.
            Give your answer as just "yes" or "no" with no extra commentary.
            
            Answer:""")
            
            chain = LLMChain(llm=self.gemini, prompt=validation_prompt)
            response = chain.invoke({
                "query": query,
                "chat_history": recent_exchanges
            })
        else:
            # Standalone query validation without history
            validation_prompt = ChatPromptTemplate.from_template("""
            You are a Movie Query Validator. Determine if the following query is related to movies or TV series. 
            It can be a question, a request for information, or a general inquiry.
            It can contain titles, actors, directors, or any other relevant details.
            Return "yes" if it is, or "no" if it is not, with no extra commentary.

            Query: {query}

            Answer:""")
            
            chain = LLMChain(llm=self.gemini, prompt=validation_prompt)
            response = chain.invoke({"query": query})
        
        answer = response['text'].strip().lower()
        return answer.startswith("yes")
    
    def process_query(self, query: str, chat_history: List[str] = None) -> Dict[str, Any]:
        """Process a user query through the integrated system"""
        if chat_history is None:
            chat_history = []

        if chat_history:
            query = self.rewrite_followup_query(query, chat_history)
        
        if not self.validate_query(query, chat_history):
            return {
                "response": "The query does not appear to be related to movies or TV series. Please ask a relevant question.",
                "sources": []
            }
        
        initial_state = MovieQueryState(
            original_query=query,
            chat_history=chat_history
        )

        try:
            result = self.agent.invoke(initial_state)
            
            # Update chat history
            chat_history.append(f"User: {query}")
            chat_history.append(f"Bot: {result['final_response']}")
            
            # Limit chat history size
            if len(chat_history) > self.chat_history_limit * 2:  
                chat_history = chat_history[-(self.chat_history_limit * 2):]
            
            return {
                "response": result["final_response"],
                "sources": result["sources"]
            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                "response": error_msg,
                "sources": []
            }



pickle_path = "/content/drive/MyDrive/content_descriptions.pkl"
chroma_db_path = "/content/drive/MyDrive/chroma_db/plots"
metadata_path = "/content/drive/MyDrive/movie_qa.pkl"
csv_path = "/content/drive/MyDrive/plot.csv"
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:

    qa_system = MovieQASystem(
        persist_directory=chroma_db_path,
        gemini_api_key=GEMINI_API_KEY,
        groq_api_key=GROQ_API_KEY
    )
    qa_system.load_metadata(metadata_path)

    collection = qa_system.chroma_client.get_collection("movie_titles")
    if collection.count() == 0:
        qa_system.load_csv_data(csv_path)
        qa_system.save_metadata(metadata_path)

    if os.path.exists(pickle_path):
        qa_system.load_pickle_data(pickle_path)

    movie_system = IntegratedMovieSystem(
        movie_qa_system=qa_system,
        gemini_api_key=GEMINI_API_KEY,
        groq_api_key=GROQ_API_KEY,
        chat_history_limit=3
    )

    print("Movie Information System")
    print("-----------------------")
    chat_memory = []
    
    while True:
        query = input("\nEnter your movie question (or 'q' to quit, 'clear' to reset conversation): ")
        
        if query.lower() == 'q':
            break
            
        if query.lower() == 'clear':
            chat_memory = []
            print("Chat history cleared.")
            continue
        
        print("\nProcessing your query...")
        result = movie_system.process_query(query, chat_history=chat_memory)
        
        print("\nRESPONSE:")
        print(f"A: {result['response']}")
        
        if result['sources']:
            print("\nSources:")
            for source in result['sources']:
                print(f"- {source.get('title', 'Unknown')}: {source.get('url', 'No URL')}")
        
except Exception as e:
    print(f"Application error: {e}")