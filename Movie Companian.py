import os
from typing import Dict, Any
from agent import IntegratedMovieSystem
from QNA import MovieQASystem
from recommender import RAGContentRecommender

class MovieAgent:
    def __init__(self, 
                gemini_api_key: str,
                groq_api_key: str,
                qa_chroma_path: str = None,     
                recommender_chroma_path: str = None, 
                metadata_path: str = None,
                csv_path: str = None,
                movies_path: str = None,
                tv_shows_path: str = None,
                descriptions_cache_path: str = None,
                chat_history_limit: int = 7):

        self.chat_history_limit = chat_history_limit
        self._initialize_systems(
            qa_chroma_path,         
            recommender_chroma_path,
            metadata_path, 
            csv_path, 
            movies_path,
            tv_shows_path,
            descriptions_cache_path,
            gemini_api_key, 
            groq_api_key
        )
        self.chat_memory = []

    def _initialize_systems(self, qa_chroma_path, recommender_chroma_path, metadata_path, 
                          csv_path, movies_path, tv_shows_path, descriptions_cache_path,
                          gemini_api_key, groq_api_key):

        # Initialize QA System
        self.qa_system = MovieQASystem(
            chroma_persist_directory=qa_chroma_path,  
            gemini_api_key=gemini_api_key,
            groq_api_key=groq_api_key
        )

        if metadata_path:
            self.qa_system.load_metadata(metadata_path)

        if qa_chroma_path: 
            collection = self.qa_system.chroma_client.get_collection("movie_titles")
            if collection.count() == 0 and csv_path:
                self.qa_system.load_csv_data(csv_path)
                if metadata_path:
                    self.qa_system.save_metadata(metadata_path)

        # Initialize Recommender System
        self.recommender = RAGContentRecommender(
            movies_path=movies_path,
            tv_shows_path=tv_shows_path,
            groq_api_key=groq_api_key,
            chroma_persist_directory=recommender_chroma_path, 
            descriptions_cache_path=descriptions_cache_path
        )
        self.movie_system = IntegratedMovieSystem(
            movie_qa_system=self.qa_system,
            gemini_api_key=gemini_api_key,
            groq_api_key=groq_api_key,
            chat_history_limit=7
        )
        
        # Build recommender index if needed
        stats = self.recommender.get_collection_stats()
        if stats.get("total_items", 0) == 0:
            print("Building recommender index...")
            self.recommender.build_index()
        else:
            print(f"Using existing recommender index with {stats.get('total_items')} items.")

    def query(self, query: str) -> Dict[str, Any]:
        
        if query.lower() == 'clear':
            self.chat_memory = []
            return {"response": "Chat history cleared.", "sources": []}
            
        result = self.movie_system.process_query(query, chat_history=self.chat_memory)
        return result

    def recommend(self, query: str, top_k: int = 5) -> str:
        """
        Get movie/TV show recommendations
        """
        return self.recommender.recommend(query, top_k=top_k)

    def start_cli(self):
        """Start an interactive CLI session"""
        print("Movie Information System")
        print("-----------------------")
        print("Commands: ")
        print("- 'q' to quit")
        print("- 'clear' to reset conversation")
        print("- 'recommend' to switch to recommendation mode")
        print("- 'qa' to switch to Q&A mode")
        
        mode = "qa"  # Default mode
        
        while True:
            if mode == "qa":
                prompt = "\nEnter your movie question"
            else:
                prompt = "\nEnter your recommendation query"
                
            query = input(f"{prompt} (or 'q' to quit, 'clear' to reset, 'recommend'/'qa' to switch modes): ")
            
            if query.lower() == 'q':
                break
            elif query.lower() == 'recommend':
                mode = "recommend"
                print("\nSwitched to recommendation mode")
                continue
            elif query.lower() == 'qa':
                mode = "qa"
                print("\nSwitched to Q&A mode")
                continue
                
            if mode == "qa":
                result = self.query(query)
                print("\nRESPONSE:")
                print(f"A: {result['response']}")
                
                if result['sources']:
                    print("\nSources:")
                    for source in result['sources']:
                        print(f"- {source.get('title', 'Unknown')}: {source.get('url', 'No URL')}")
            else:
                recommendations = self.recommend(query)
                print("\nRECOMMENDATIONS:")
                print(recommendations)


agent = MovieAgent(
    gemini_api_key = os.environ.get("GOOGLE_API_KEY"),
    groq_api_key= os.environ.get("GROQ_API_KEY"),
    recommender_chroma_path="/persist/chroma_db/recommender",
    metadata_path="/path/to/your/metadata",
    descriptions_cache_path="/path/to/your/pickle",
    csv_path="/path/to/your/csv",
    qa_chroma_path="/persist/chroma_db/plots",           
    movies_path="/path/to/your/csv",
    tv_shows_path="/path/to/your/csv",
)

# Use Q&A
result = agent.query("Who directed Inception?")
print(result['response'])

# Get recommendations
recommendations = agent.recommend("movies like Inception")
print(recommendations)

# Or use the interactive CLI
agent.start_cli()

