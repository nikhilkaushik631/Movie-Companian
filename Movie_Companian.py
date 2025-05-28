import os
import asyncio
from Multi_Agent import IntegratedQASystem
from Content_Recommend import RAGContentRecommender
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()
class MovieChatbot:
    def __init__(self, 
                google_api_key: str = None,
                groq_api_key: str = None,
                chat_history_limit: int = 15):
            
        self.google_api_key = google_api_key
        self.groq_api_key = groq_api_key

        self.chat_history_limit = chat_history_limit

        self.recommender = RAGContentRecommender()
        print("RAGContentRecommender initialized.")

        self.movie_system = IntegratedQASystem()
        print("IntegratedMovieSystem initialized.")
        print("All core systems initialized.")
        self.chat_memory = []

        self._initialize_langchain()
 
    def _initialize_langchain(self):
        """Initialize LangChain components for conversation handling"""

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=self.google_api_key,
            temperature=0.7
        )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=self.groq_api_key,
            temperature=0.2 
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=self.chat_history_limit
        )

        validation_prompt_with_history = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a Movie Query Validator. Based on the recent conversation and the current query,
                determine if the current query is related to movies or TV series.
                This could be a follow-up question to the previous movie conversation.
                Consider both the query itself and its context in the conversation.
                Return "yes" if it's related to movies/TV (either directly or as a follow-up), or "no" if it's not.
                Give your answer as just "yes" or "no" with no extra commentary."""
            ),
            MessagesPlaceholder(variable_name="chat_history"), 
            HumanMessagePromptTemplate.from_template("Current query: {query}\n\nAnswer:")
        ])

        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=validation_prompt_with_history,
            memory=self.memory, 
            output_key="validation_result"
        )

        router_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a Movie Query Router that determines whether a user query is asking for:
                1. Information/facts about movies or TV shows (return "qa")
                2. Movie or TV show recommendations (return "recommend")

                Return ONLY "qa" or "recommend" with no additional text or explanation."""
                        ),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{query}")
                    ])
        self.router_chain = LLMChain(
            llm=self.llm,
            prompt=router_prompt,
            memory=self.memory, 
            output_key="route"
        )


        rewriting_prompt_messages = [
            SystemMessagePromptTemplate.from_template(
                """You are an expert query rewriter for a movie and TV information system. Your task is to reformulate the followup query so that it is self-contained and includes all necessary context from the conversation history to be answered properly.

                    INSTRUCTIONS:
                    1. Identify all implicit references in the query (pronouns, implied subjects, contextual terms).
                    2. Determine the specific information being requested (facts, lists, comparisons, explanations).
                    3. Incorporate ALL relevant context from the conversation history.
                    4. For requests seeking detailed information, explicitly include that in the rewrite.
                    5. Ensure the rewritten query can stand alone and conveys the full intent.
                    If the query is already self-contained and requires no context from history, or if there is no history, return the original query as is.
                    """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                """CURRENT FOLLOW-UP QUERY:
                    {query}

                    REWRITTEN QUERY (make it detailed, explicit, and comprehensive):"""
                            )
                        ]
        rewriting_prompt = ChatPromptTemplate.from_messages(rewriting_prompt_messages)

        self.rewriting_chain = LLMChain(
            llm=self.gemini,
            prompt=rewriting_prompt,
            memory=self.memory,
            output_key="rewritten_query"
        )
        print("LangChain components initialized.")

    async def validate_movie_query(self, query: str) -> bool:
        """
        Validate if the query is related to movies or TV shows
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.validation_chain.invoke, {"query": query})
            validation_result = result.get("validation_result", "").strip().lower()
            return validation_result == "yes"
        except Exception as e:
            print(f"Error in validation: {e}")
            return False

    async def rewrite_query(self, query: str) -> str:
        """
        Rewrite the query to be self-contained if chat history exists.
        Uses the enhanced rewriting prompt.
        """
        try:
            current_history = self.memory.load_memory_variables({}).get("chat_history", [])
            if not current_history:
                return query 

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.rewriting_chain.invoke, {"query": query})
            rewritten_query = result.get("rewritten_query", "").strip()
            return rewritten_query if rewritten_query else query
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query 

    async def route_query(self, query: str) -> str:
        """
        Determine whether the query is for QA or recommendation
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.router_chain.invoke, {"query": query})
            route = result.get("route", "").strip().lower()
            return route if route in ["qa", "recommend"] else "qa"
        except Exception as e:
            print(f"Error routing query: {e}")
            return "qa"

    async def chat(self, user_input: str, session_id: str = None) -> str:
        """
        Process a user message and return a conversational response
        """
   
        if user_input.lower() == 'clear':
            self.memory.clear()
            return "Our conversation has been cleared. What would you like to know about movies or TV shows?"

        if user_input.lower() == 'q':
            return "Goodbye! Enjoy your movie watching!"

        is_movie_related = await self.validate_movie_query(user_input)

        if not is_movie_related:
            self.memory.save_context(
                {"input": user_input},
                {"output": "I'm specialized in movies and TV shows. Could you ask me something related to films, series, actors, directors, or similar topics?"}
            )
            return "I'm specialized in movies and TV shows. Could you ask me something related to films, series, actors, directors, or similar topics?"

        resolved_query = await self.rewrite_query(user_input)

        query_type = await self.route_query(resolved_query)

        response_text = ""
        try:
            if query_type == "qa":
                if hasattr(self.movie_system, 'aprocess_query'):
                     result = await self.movie_system.aprocess_query(resolved_query)
                else:
                    result = await self.movie_system.process_query(resolved_query)

                response_text = result.get('response', 'I could not find information about that.')
            else:  
                if hasattr(self.recommender, 'arecommend'):
                    response_text = await self.recommender.arecommend(resolved_query, top_k=5)
                else:
                    response_text = await self.recommender.recommend(resolved_query, 5)
                    
        except Exception as e:
            print(f"Error during QA or Recommendation processing: {e}")
            response_text = "Sorry, I encountered an issue while processing your request."

        self.memory.save_context({"input": user_input}, {"output": response_text})

        return response_text


#     async def start_chatbot_cli(self):
#         """Start an interactive chatbot CLI session"""
#         print("MovieBot")
#         print("------------------------------------------------")
#         print("Ask me anything about movies and TV shows, or ask for recommendations!")
#         print("Type 'q' to quit or 'clear' to reset our conversation.")

#         while True:
#             try:
#                 user_input = await asyncio.to_thread(input, "\nYou: ")     
#             except KeyboardInterrupt:
#                 print("\nMovieBot: Exiting...")
#                 break

#             if user_input.lower() == 'q':
#                 print("\nMovieBot: Goodbye! Enjoy your movie watching!")
#                 break

#             response = await self.chat(user_input)
#             print(f"\nMovieBot: {response}")



# async def main_async():
#     """Asynchronous main function to run the chatbot."""
#     chatbot = MovieChatbot(
#         google_api_key = os.getenv("google_api_key"),
#         groq_api_key = os.getenv("groq_api_key")
#     )

#     # queries_to_test = [
#     #     # "Who directed Inception?",
#     #     # "Did he direct any other sci-fi movies?",
#     #     # "Recommend me movies like those",
#     #     # "What is the capital of France?",
#     #     # "What is the capital of Italy?",
#     #     # "movies with great cgi like dune part two",
#     #     "who directed dune part two?",
#     #     "japanese anime like bleach",
#     #     # "movies like interstellar",
#     #     # "In the movie 'The Shawshank Redemption', who is the main character?",
#     #     # "What happens at the end of Inception?",
#     #     # "Tell me about the plot of 'The Godfather'",
#     # ]

#     # for query in queries_to_test:
#     #     print(f"\nUser: {query}") 
#     #     response = await chatbot.chat(query)
#     #     print(f"MovieBot: {response}") 
#     #     await asyncio.sleep(2)

#     await chatbot.start_chatbot_cli()

# if __name__ == "__main__":
#     # To run the async version:
#     try:
#         #  import nest_asyncio
#         #  nest_asyncio.apply()
#         #  asyncio.run(main_async())
#         asyncio.run(main_async())
#     except KeyboardInterrupt:
#         print("Exited by user.")