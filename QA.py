import os
import pickle
import re
import string
import warnings
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
    A question-answering system that combines vector similarity search with LLM-enhanced
    responses about movies and TV shows. Uses Chroma as the vector database.
    """

    def __init__(
        self,
        csv_path=os.environ.get("plots_path"),
        additional_data_path=os.environ.get("description_path"),
        # embedding_model_name: str = "all-MPNet-base-v2",
        embedding_model_name: str = "all-MiniLM-L6-V2",
        google_api_key=os.environ.get("google_api_key"),
        groq_api_key=os.environ.get("groq_api_key"),
        pinecone_api_key=os.environ.get("pinecone_api_key"),
        # pinecone_index_name: str = "movie-titles",
        pinecone_index_name: str = "movie-plot",
        use_memory_cache: bool = True,
        metadata_cache_path=os.environ.get("metadata_path")
    ):
        self.additional_data_path = additional_data_path
        self.use_memory_cache = use_memory_cache
        self.metadata_cache_path = metadata_cache_path
        self.csv_path = csv_path
        self.pickle_data = None
        self.title_to_imdb = {}
        self._query_cache = {}

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
            # print("Index Stats: ",self.index.describe_index_stats()) # Print stats for confirmation

        except Exception as e:
            raise RuntimeError(f"Could not initialize Pinecone: {e}") from e

        self._load_metadata()
        self.load_pickle_data()
        # stats = self.verify_pinecone_collection()
        # print(f"Index Stats: {stats}")
        self.plots_df = pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else None
        self.groq = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )
        self.groq1 = ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key
        )
        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
        )


    def _load_metadata(self):
        """
        Load metadata from disk if available.
        """
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
        """
        Verify the existing ChromaDB collection and its contents.
        Returns True if collection exists and has data, False otherwise.
        """
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
        """
        Load additional movie data from pickle file
        """
        path_to_use = pickle_path or self.additional_data_path

        try:
            with open(path_to_use, 'rb') as f:
                self.pickle_data = pickle.load(f)
        except Exception as e:
            self.pickle_data = {}


    async def find_best_title_match(self, query: str, top_k: int = 3):
        """
        Find the best matching titles for a query using pinecone vector search.
        """
        if not hasattr(self, 'index') or self.index is None:
            print("Error: Pinecone index not initialized.")
            return []

        try:

            query_vector = await asyncio.get_event_loop().run_in_executor(
                None,
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
        return matches
    

    async def get_movie_data_by_title(self, title_query):
        """
        Get plot and Wikipedia introduction for a specific title using multi-strategy approach
        """
        if not hasattr(self, 'index') or self.index is None:
            return None, None, None
        
        def _normalize_title(title: str) -> str:
            """
            Normalize a title string by converting to lowercase and removing punctuation.
            """
            if not title:
                return ""
            title = title.translate(str.maketrans('', '', string.punctuation))
            return ' '.join(title.lower().split())
                            
        # 1. Try a normalized match using top vector search results
        try:
            norm_query_title = _normalize_title(title_query)
            potential_matches = await self.find_best_title_match(title_query, top_k=5)

            for p_title, p_metadata, p_similarity in potential_matches:
                # Check for exact normalized match first
                if self._normalize_title(p_title) == norm_query_title:
                    plot = p_metadata.get('plot')
                    wiki_intro = p_metadata.get('wikipedia_intro')
                    imdb_id = p_metadata.get('imdb_id')
                    # Pinecone metadata might need adjustment based on how you populated it
                    if plot or wiki_intro:
                        return (plot, wiki_intro, imdb_id)
        except Exception as e:
            pass


        # 2. Fallback to DataFrame search if self.plots_df is populated and relevant
        if self.plots_df is not None and not self.plots_df.empty:
            try:
                df_matches = self.plots_df[self.plots_df['title'].str.lower() == title_query.lower()]
                if not df_matches.empty:
                    row = df_matches.iloc[0]
                    plot = row.get('plot')
                    wiki_intro = row.get('wikipedia_intro')
                    imdb_id = row.get('imdb_id')
                    if plot or wiki_intro:
                        return (plot, wiki_intro, imdb_id)
            except (KeyError, Exception) as e:
                pass

        # 3. If still not found, use the best similarity match above a threshold
        if potential_matches and potential_matches[0][2] > 0.7:
            best_match_meta = potential_matches[0][1]
            plot = best_match_meta.get('plot')
            wiki_intro = best_match_meta.get('wikipedia_intro')
            imdb_id = best_match_meta.get('imdb_id')
            if plot or wiki_intro:
                return (plot, wiki_intro, imdb_id)

        return None, None, None
    

    def get_movie_info_by_imdb_id(self, imdb_id):
        """
        Get additional movie information from pickle data by IMDB ID
        """
        return self.pickle_data.get(imdb_id, None)
    

    async def _extract_title_with_llm(self, query, llm_name):
        """
        Use LLM to extract movie title from a query
        """
        llm = getattr(self, llm_name) if hasattr(self, llm_name) else None
        if not llm:
            return None

        title_prompt = ChatPromptTemplate.from_template("""
        Extract the exact movie or tv series title from this query.
        Return ONLY the movie title with no additional text, quotation marks or explanations.
        If you cannot identify a movie title with certainty, return the single word 'None'.

        Query: "{query}"

        Movie title:
        """)

        chain = LLMChain(llm=llm, prompt=title_prompt)
        # loop = asyncio.get_event_loop()
        # response = await loop.run_in_executor(None, chain.invoke, {"query": query})
        response = await chain.ainvoke({"query": query})
        extracted_title = response['text'].strip()

        if extracted_title.lower() == 'none':
            return None

        return extracted_title
    
    async def extract_title_from_query(self, query):
        """
        Extract potential movie title from query using semantically enhanced title database
        """
        extracted_title = None
        
        # Try with Gemini first if available
        if self.gemini:
            extracted_title = await self._extract_title_with_llm(query, 'gemini')
        
        # Fallback to another LLM 
        if not extracted_title and self.groq:
             extracted_title = await self._extract_title_with_llm(query, 'groq')

        if extracted_title:
            # Validate if this title exists in our DB (plot/wiki needed)
            movie_data = await self.get_movie_data_by_title(extracted_title)
            plot, wiki_intro, _ = movie_data            
            if plot or wiki_intro:
                try:
                    pattern = re.compile(r'\b' + re.escape(extracted_title) + r'\b', re.IGNORECASE)
                    clean_query = pattern.sub('', query, count=1) 
                    if clean_query != query:
                        clean_query = ' '.join(clean_query.split())
                    else:
                        clean_query = query 
                except re.error:
                    clean_query = query
                
                return (extracted_title, clean_query if clean_query else query)
            else:
                extracted_title = None

        # Fallback to vector search if LLM fails or title not validated
        matches = await self.find_best_title_match(query, top_k=3)
        if matches:
            best_match_title, _, best_similarity = matches[0]
            if best_similarity > 0.65:
                plot_vs, wiki_intro_vs, _ = await self.get_movie_data_by_title(best_match_title)
                if plot_vs or wiki_intro_vs:
                    try:
                        pattern = re.compile(r'\b' + re.escape(best_match_title) + r'\b', re.IGNORECASE)
                        clean_query = pattern.sub('', query, count=1)
                        clean_query = ' '.join(clean_query.split()) if clean_query != query else query
                    except re.error:
                        clean_query = query

                    return (best_match_title, clean_query if clean_query else query)

        return (None, query)

    async def answer_query(self, query):
        """
        Answer a query about a movie using detailed, structured context with LLM.
        Now leveraging both plot and Wikipedia introduction data directly.
        """
        title, actual_query = await self.extract_title_from_query(query)
        if not title:
            matches = await self.find_best_title_match(query, top_k=3)

            if matches and matches[0][2] > 0.75: 
                title = matches[0][0]
                metadata = matches[0][1]
                actual_query = query
            else:
                return "I couldn't identify a movie in your question. Please ensure you include the movie name or specific details about the plot."

        mo = await self.get_movie_data_by_title(title)
        plot, wiki_intro, imdb_id = mo
        if not plot and not wiki_intro:
            return f"I couldn't find information about '{title}'. Please check the movie title."

        additional_info = None
        if imdb_id:
            additional_info = self.get_movie_info_by_imdb_id(imdb_id)

        context_lines = [
            f"Movie: {title}",
        ]

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
                context_lines.append(f"- Genres: {', '.join(content['genres']) if isinstance(content['genres'], (list, tuple)) else content['genres']}")
            if people.get('cast'):
                context_lines.append(f"- Main Cast: {', '.join(people['cast'][:5]) if isinstance(people['cast'], (list, tuple)) else people['cast']}")
            if 'movie_specific' in additional_info:
                movie_info = additional_info['movie_specific']
                if movie_info.get('release_date'):
                    context_lines.append(f"- Release Date: {movie_info['release_date']}")
                if movie_info.get('directors'):
                    context_lines.append(f"- Director(s): {', '.join(movie_info['directors']) if isinstance(movie_info['directors'], (list, tuple)) else movie_info['directors']}")
            elif 'tv_specific' in additional_info:
                tv_info = additional_info['tv_specific']
                if tv_info.get('seasons_count'):
                    context_lines.append(f"- Seasons: {tv_info['seasons_count']}")
                if tv_info.get('air_years'):
                    context_lines.append(f"- Air Years: {tv_info['air_years']}")

        context = "\n".join(context_lines)

        if len(context_lines) < 3: 
            return f"I don't have enough information about '{title}' to answer your question properly."

        meta_prompt = ChatPromptTemplate.from_template("""
            You are a knowledgeable and engaging Movie Companion AI.
            Based on the provided context information about a movie or TV series, answer the user's question.

            CONTEXT INFORMATION:
            {context}

            USER'S QUESTION (about the movie/series mentioned above): "{user_question}"

            INSTRUCTIONS:
            1.  Carefully analyze the user's question to understand what they are asking for (e.g., plot details, cast, release year, opinion on a specific aspect based on the plot, etc.).
            2.  Use **all relevant information** from the CONTEXT (plot, Wikipedia intro, additional details) if required to formulate your answer.
            3.  If the question is about the story, plot, narrative, premise, or synopsis, provide an engaging story like a storyteller would based on the available plot and introduction. 
            4.  If the question is about specific facts (e.g., director, cast, release date, genre), provide those details directly from the context.
            5.  If the context includes a Wikipedia introduction and a plot summary, synthesize information from both if they complement each other for a richer answer.
            6.  Address the user's question directly and comprehensively. The questions which require one word answers, only provide that no need for extra explanation.
            7.  Be informative and try to sound natural and conversational. try to create the answers in first person. Do no include json or code blocks in your response.
            8.  If the context is insufficient to answer the specific question, clearly state that you don't have that specific piece of information in the provided details for '{movie_title}'. Do not invent information.
            9.  If the user's question is very vague (e.g., "tell me about it"), provide a general overview including plot, genre, and key available details.


            YOUR RESPONSE:
            """)

        try:
            chain = LLMChain(llm=self.groq1, prompt=meta_prompt)
            # loop = asyncio.get_event_loop()
            # response = await loop.run_in_executor(None, chain.invoke, {
            #     "context": context,
            #     "user_question": actual_query if actual_query else query,
            #     "movie_title": title
            # })
            response = await chain.ainvoke({
                "context": context,
                "user_question": actual_query if actual_query else query,
                "movie_title": title
            })
            return response['text'].strip()

        except Exception as e:
            return f"I encountered an error while trying to answer your question: {str(e)}"


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