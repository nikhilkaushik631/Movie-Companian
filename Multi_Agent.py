import os
import warnings
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import json
from QA import MovieQASystem
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import asyncio
import aiohttp
from dotenv import load_dotenv
load_dotenv()
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

class IntegratedQASystem:
    """Movie information system using LangGraph for agent orchestration"""
    
    def __init__(
        self,
        gemini_api_key=os.environ.get("gemini_api_key"),
        groq_api_key=os.environ.get("groq_api_key"),
        chat_history_limit: int = 10
    ):
        self.qa_system = MovieQASystem()
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
    
    async def _query_database_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for querying the existing movie database"""
        try:
            db_response = await self.qa_system.answer_query(state.original_query)
            # print(f"Database response: {db_response}")  
            state.db_response = db_response
        except Exception as e:
            state.db_response = f"Error querying database: {str(e)}"
        
        return state

    async def _analyze_query_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for analyzing if web search is needed"""

        analysis_prompt = ChatPromptTemplate.from_template("""
        You are a highly specialized Query Analyzer Agent for a Movie & TV Series information system.
        Your task is to analyze the User Query and the Database Response to decide if a Web Search is necessary.

        **User Query:** {original_query}

        **Database Response:** {db_response}

        **Instructions:**

        1.  **Evaluate:** Does the Database Response *fully* and *specifically* answer the User Query?
            * Consider if the query asks for:
                * **Recent/Timely Info:** Latest releases, news, current streaming availability, reviews (Likely needs web search).
                * **Subjective Info:** Rankings, opinions, "best" lists (Likely needs web search).
                * **Title Mismatch:** The embeddings may not have the exact title or recent info, so check if the title in the query matches the title in the DB response.
                * **Specific Facts:** Dates, names, plot points (Check if DB provided them).
        2.  **Short Answers:** Most of the queries related to cast, crew, characters, short infos and other specific information can be answered by the database if the title matches.
        3.  **Identify Gaps:** Pinpoint *exactly* what information is missing or insufficient.
        4.  **Decide:** Set `needs_web_search` to `true` if the DB response is insufficient based on the evaluation, otherwise set it to `false`.
        5.  **Formulate Search Query:** If `needs_web_search` is `true`, create a concise, effective search query targeting *only* the missing information. If not needed, leave it empty.

        **Output Format:**
        Your response MUST be a single, valid JSON object. Do NOT include any text, explanations, or markdown before or after the JSON block.

        ```json
        {{
        "needs_web_search": boolean,
        "reason": "A brief explanation of why web search is needed, or why not.",
        "entities": ["list", "of", "key", "entities"],
        "search_query": "Optimized search query (or empty string if not needed)."
        }}
        """)
        
        chain = LLMChain(llm=self.groq1, prompt=analysis_prompt)
        response = await chain.ainvoke({
            "original_query": state.original_query,
            "db_response": state.db_response
        })
        
        try:
            # Clean response and extract JSON
            response_text = response['text'].strip()
            
            # Find JSON object in response
            import re
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response_text
                
            result = json.loads(json_str)
            state.needs_web_search = result.get("needs_web_search", False)
            state.search_query = result.get("search_query", state.original_query + " movie info")
            state.analysis_result = result
            if state.needs_web_search:
                print("I couldn't find the answer in my database. Please wait while I search the web for you...")
            
        except Exception as e:
            # General fallback: Be cautious and trigger a search.
            state.needs_web_search = True
            state.search_query = f"{state.original_query} movie tv series info"
            state.analysis_result = {
                "needs_web_search": True,
                "entities": [e.strip() for e in state.original_query.split() if len(e) > 3],
                "search_query": state.search_query,
                "reason": f"Analysis fallback due to an unexpected error: {e}"
            }
        
        return state
    

    def _route_based_on_analysis(self, state: MovieQueryState) -> str:
        """Determine the next node based on analysis results"""
        if state.needs_web_search:
            print("Searching Web")
            return "needs_web_search"
        else:
            print("Answering from Embeddings")
            return "sufficient_info"
    

    async def _web_search_agent(self, state: MovieQueryState) -> MovieQueryState:
        """Agent for performing web searches using DuckDuckGo"""
        search_results = {}
        
        try:
            print(f"Searching DuckDuckGo for: {state.search_query}")
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(
                    state.search_query, 
                    region='wt-wt', 
                    safesearch='Moderate', 
                    max_results=2
                )]
            
            urls = [r["href"] for r in results]

            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        print(f"Scraping: {url}")
                        async with session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}) as response:
                            content = await response.text()
                            
                        soup = BeautifulSoup(content, "html.parser")
                        text_content = soup.get_text(separator="\n", strip=True)
                        
                        # Limit content size
                        if len(text_content) > 7000:
                            text_content = text_content[:7000] + "... [content truncated]"
                        
                        result = {
                            "url": url,
                            "title": soup.title.string if soup.title else "Unknown",
                            "content": text_content,
                            "query": state.search_query,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        search_results[url] = result
                    except Exception as e:
                        print(f"Error scraping {url}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Web search error: {e}")
        
        state.raw_search_results = search_results
        return state

    async def _content_extraction_agent(self, state: MovieQueryState) -> MovieQueryState:
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

        
        chain = LLMChain(llm=self.groq2, prompt=content_extraction_prompt)
        # loop = asyncio.get_event_loop()
        # response = await loop.run_in_executor(None, chain.invoke, {
        #     "original_query": state.original_query,
        #     "search_query": state.search_query,
        #     "reason": state.analysis_result.get("reason", "Information gathering"),
        #     "raw_search_results": formatted_results
        # })
        response = await chain.ainvoke({
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
    

    async def _response_synthesis_agent(self, state: MovieQueryState) -> MovieQueryState:
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
        
        chain = LLMChain(llm=self.gemini, prompt=response_synthesis_prompt)
        # loop = asyncio.get_event_loop()
        # response = await loop.run_in_executor(None, chain.invoke, {
        #     "original_query": state.original_query,
        #     "db_response": state.db_response,
        #     "web_search_info": web_search_info,
        #     "chat_history": "\n".join(state.chat_history) if state.chat_history else "None"
        # })
        
        response = await chain.ainvoke({
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


    
    async def process_query(self, query: str, chat_history: List[str] = None) -> Dict[str, Any]:
        """Process a user query through the integrated system"""
        if chat_history is None:
            chat_history = []

        
        initial_state = MovieQueryState(
            original_query=query,
            chat_history=chat_history
        )

        try:
            # loop = asyncio.get_event_loop()
            # result = await loop.run_in_executor(None, self.agent.invoke, initial_state)
            result = await self.agent.ainvoke(initial_state)
            
            # Update chat history
            chat_history.append(f"User: {query}")
            chat_history.append(f"Bot: {result['final_response']}")
            
            # Limit chat history size
            if len(chat_history) > self.chat_history_limit * 2:  
                chat_history = chat_history[-(self.chat_history_limit * 2):]
            
            return {
                "response": result["final_response"],
                "sources": result["sources"],
                "used_web_search": result.get("analysis_result", {}).get("needs_web_search", False)

            }
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                "response": error_msg,
                "sources": []
            }



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