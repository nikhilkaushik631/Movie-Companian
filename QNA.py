import os
import pickle
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import logging
import string
import chromadb
from chromadb.utils import embedding_functions
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovieQASystem:
    """
    A question-answering system that combines vector similarity search with LLM-enhanced
    responses about movies and TV shows. Uses Chroma as the vector database.
    """

    def __init__(
        self,
        plots_path: str = "/content/drive/MyDrive/plots.csv",
        additional_data_path: str = "/content/drive/MyDrive/content_descriptions.pkl",
        embedding_model_name: str = "all-MPNet-base-v2",
        gemini_api_key: str = None,
        groq_api_key: str = None,
        chroma_persist_directory: str = "/content/drive/MyDrive/chroma_db/plots",
        use_memory_cache: bool = True,
        metadata_cache_path: str = "/content/drive/MyDrive/movie_qa_metadata.pkl"
    ):

        self.plots_path = plots_path
        self.additional_data_path = additional_data_path
        self.chroma_persist_directory = chroma_persist_directory
        self.use_memory_cache = use_memory_cache
        self.metadata_cache_path = metadata_cache_path

        self.plots_df = None
        self.pickle_data = None
        self.title_to_imdb = {}
        self._query_cache = {}

        os.makedirs(chroma_persist_directory, exist_ok=True)

        logger.info(f"Setting up embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        try:
            from chromadb.config import Settings
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                    persist_directory=chroma_persist_directory,
                )
            )
        except Exception as e:
            logger.error(f"Warning: Failed to initialize ChromaDB with custom settings: {e}")
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)

        self._initialize_collection()
        self._load_metadata()

        self.groq = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
        )

        if plots_path and os.path.exists(plots_path):
            self.load_csv_data(plots_path)

        if additional_data_path and os.path.exists(additional_data_path):
            self.load_pickle_data(additional_data_path)

    def _initialize_collection(self):
        """
        Initialize the ChromaDB collection with proper error handling.
        """
        collection_name = "movie_titles"

        # First try to get the existing collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Successfully connected to existing collection '{collection_name}'.")
            return
        except Exception as e:
            logger.info(f"Did not find existing collection: {e}")

        # If retrieving fails, try to create a new collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Successfully created new collection '{collection_name}'.")
            return
        except Exception as e:
            logger.error(f"Failed to create new collection: {e}")

        # If both attempts fail, check if we have any collections at all
        try:
            collections = self.chroma_client.list_collections()
            if collections:
                logger.info(f"Found existing collections: {[c.name for c in collections]}")

                # Try to reset collection as a last resort
                try:
                    logger.info(f"Attempting to reset collection '{collection_name}'...")
                    self.chroma_client.delete_collection(collection_name)
                    self.collection = self.chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Successfully reset and created collection '{collection_name}'.")
                    return
                except Exception as reset_error:
                    logger.error(f"Failed to reset collection: {reset_error}")
            else:
                logger.warning("No collections found in the database.")
        except Exception as list_error:
            logger.error(f"Failed to list collections: {list_error}")

        raise RuntimeError("Failed to initialize ChromaDB collection after multiple attempts. Check your database configuration.")

    def _save_metadata(self):
        """
        Save metadata to disk.
        """
        metadata = {
            'title_to_imdb': self.title_to_imdb
        }

        try:
            with open(self.metadata_cache_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved metadata to {self.metadata_cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False

    def _load_metadata(self):
        """
        Load metadata from disk if available.
        """
        if os.path.exists(self.metadata_cache_path):
            try:
                with open(self.metadata_cache_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.title_to_imdb = metadata.get('title_to_imdb', {})
                logger.info(f"Loaded metadata from {self.metadata_cache_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.title_to_imdb = {}
                return False
        else:
            logger.info("No saved metadata found.")
            self.title_to_imdb = {}
            return False

    def load_csv_data(self, csv_path=None):
        """
        Load plot data from CSV file and store in ChromaDB.
        """
        path_to_use = csv_path or self.plots_path

        try:
            logger.info(f"Loading CSV data from {path_to_use}")
            self.plots_df = pd.read_csv(path_to_use)
            logger.info(f"Loaded {len(self.plots_df)} records from CSV")

            self.title_to_imdb = dict(zip(self.plots_df['title'], self.plots_df['imdb_id']))
            if self.collection.count() == 0:
                self._process_plots_batch()

            self._save_metadata()

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise

    def _process_plots_batch(self):
        """
        Process all movie plots in batches for better performance and add to ChromaDB.
        Using plot and wikipedia_intro for embeddings along with title.
        """
        logger.info("Processing movie plots in batches with enhanced embeddings...")

        BATCH_SIZE = 256

        total_items = len(self.plots_df)
        batch_count = (total_items + BATCH_SIZE - 1) // BATCH_SIZE

        def prepare_text(row):
            title_text = row.get('title', 'Unknown')
            plot_text = row.get('plot', '')
            wiki_text = row.get('wikipedia_intro', '')
            combined_text = f"Title: {title_text}"
            if plot_text:
                combined_text += f" Plot: {plot_text}"
            if wiki_text:
                combined_text += f" Wikipedia: {wiki_text}"
            return combined_text

        for batch_idx in range(batch_count):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_items)
            batch_df = self.plots_df.iloc[start_idx:end_idx]

            batch_ids = []
            batch_texts = []
            batch_metadatas = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                combined_texts = list(executor.map(prepare_text, [row for _, row in batch_df.iterrows()]))

            for i, (_, row) in enumerate(batch_df.iterrows()):
                try:
                    content_id = str(start_idx + i)
                    metadata = {
                        "imdb_id": row.get('imdb_id'),
                        "title": row.get('title', 'Unknown'),
                        "plot": row.get('plot', ''),
                        "wikipedia_intro": row.get('wikipedia_intro', '')
                    }
                    batch_ids.append(content_id)
                    batch_texts.append(combined_texts[i])
                    batch_metadatas.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing item {start_idx + i}: {e}")
                    continue

            try:
                embeddings = self.embedding_model.encode(batch_texts, batch_size=64, show_progress_bar=False, device='cuda' if torch.cuda.is_available() else 'cpu')
            except Exception as e:
                logger.error(f"Error during embedding computation: {e}")
                continue

            if batch_ids:
                try:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=embeddings.tolist(),
                        documents=batch_texts,
                        metadatas=batch_metadatas
                    )
                    logger.info(f"Added batch {batch_idx+1}/{batch_count} with {len(batch_ids)} items to ChromaDB")
                except Exception as e:
                    logger.error(f"Error adding batch {batch_idx+1} to ChromaDB: {e}")

        logger.info(f"Completed embedding creation. Total items in ChromaDB: {self.collection.count()}")

    def load_pickle_data(self, pickle_path=None):
        """
        Load additional movie data from pickle file
        """
        path_to_use = pickle_path or self.additional_data_path

        try:
            logger.info(f"Loading pickle data from {path_to_use}")
            with open(path_to_use, 'rb') as f:
                self.pickle_data = pickle.load(f)
            logger.info(f"Loaded pickle data with {len(self.pickle_data)} records")
        except Exception as e:
            logger.error(f"Error loading pickle data: {e}")
            self.pickle_data = {}

    def find_best_title_match(self, query, top_k=3):
        """
        Find the best matching titles for a query using ChromaDB with enhanced semantic search
        """

        if self.use_memory_cache:
            cache_key = f"title_match_{query}_{top_k}"
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        matches = []
        for i in range(len(results['documents'][0])):
            # Extract just the title from each metadata entry
            metadata = results['metadatas'][0][i]
            title = metadata.get('title', 'Unknown')
            distance = results['distances'][0][i]
            # Convert distance to similarity score
            similarity = 1.0 - min(1.0, distance / 2.0)
            matches.append((title, metadata, similarity))

        if self.use_memory_cache:
            self._query_cache[cache_key] = matches

        return matches

    def get_movie_data_by_title(self, title):
        """
        Get plot and Wikipedia introduction for a specific title using multi-strategy approach
        """
        # 1. Check cache if enabled
        if self.use_memory_cache:
            cache_key = f"movie_data_{title}"
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

        # 2. Try exact match in ChromaDB metadata
        results = self.collection.query(
            query_texts=[title],
            n_results=10,
            where={"title": {"$eq": title}}
        )

        if results and results['metadatas'][0]:
            metadata = results['metadatas'][0][0]
            plot = metadata.get('plot')
            wiki_intro = metadata.get('wikipedia_intro')
            imdb_id = metadata.get('imdb_id')

            if plot or wiki_intro:
                result = (plot, wiki_intro, imdb_id)
                if self.use_memory_cache:
                    self._query_cache[cache_key] = result
                return result

        def normalize_title(title):
            title = title.translate(str.maketrans('', '', string.punctuation))
            return ' '.join(title.lower().split())

        # 3. Try case-insensitive match in collection
        metadatas = self.collection.get()['metadatas']
        norm_query = normalize_title(title)
        for md in metadatas:
            if normalize_title(md['title']) == norm_query:
                result = (md.get('plot'), md.get('wikipedia_intro'), md.get('imdb_id'))
                if self.use_memory_cache:
                    self._query_cache[cache_key] = result
                return result

        # 4. Fall back to DataFrame search if available
        if self.plots_df is not None:
            exact_matches = self.plots_df[self.plots_df['title'].str.lower() == title.lower()]
            if not exact_matches.empty:
                row = exact_matches.iloc[0]
                wiki_intro = row.get('wikipedia_intro') if 'wikipedia_intro' in row else None
                result = (row['plot'], wiki_intro, row['imdb_id'])
                if self.use_memory_cache:
                    self._query_cache[cache_key] = result
                return result

        # 5. As a last resort, use similarity search
        matches = self.find_best_title_match(title, top_k=1)
        if matches and matches[0][2] > 0.7:  # Similarity threshold
            metadata = matches[0][1]
            result = (metadata.get('plot'), metadata.get('wikipedia_intro'), metadata.get('imdb_id'))
            if self.use_memory_cache:
                self._query_cache[cache_key] = result
            return result

        return None, None, None

    def get_movie_info_by_imdb_id(self, imdb_id):
        """
        Get additional movie information from pickle data by IMDB ID
        """
        if self.pickle_data is None:
            logger.warning("No pickle data available. Please load pickle data first.")
            return None

        if self.use_memory_cache:
            cache_key = f"movie_info_{imdb_id}"
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

        movie_info = self.pickle_data.get(imdb_id, None)
        if self.use_memory_cache and movie_info:
            self._query_cache[cache_key] = movie_info

        return movie_info

    def extract_title_from_query(self, query):
        """
        Extract potential movie title from query using semantically enhanced title database
        """
        # Check cache first if enabled
        if self.use_memory_cache:
            cache_key = f"extract_title_{query}"
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

        try:
            # Use Gemini to extract the movie title
            title = self._extract_title_with_gemini(query)
            logger.info(f"Gemini extracted title: {title}")  

            if title:
                logger.info(f"Gemini extracted title: '{title}'")
                # Validate the title against our database
                plot, wiki_intro, imdb_id = self.get_movie_data_by_title(title)
                if plot or wiki_intro:
                    pattern = re.compile(r'\b' + re.escape(title.lower()) + r'\b', re.IGNORECASE)
                    match = pattern.search(query)
                    if match:
                        start, end = match.span()
                        clean_query = query[:start] + ' ' * (end-start) + query[end:]
                        clean_query = ' '.join(clean_query.split())
                    else:
                        clean_query = query

                    result = (title, clean_query)
                    if self.use_memory_cache:
                        self._query_cache[cache_key] = result
                    return result
        except Exception as e:
            logger.warning(f"Error using Gemini for title extraction: {e}")

        matches = self.find_best_title_match(query, top_k=5)

        if matches:
            # Using a slightly lower threshold since we now have richer embeddings
            high_similarity_matches = [m for m in matches if m[2] > 0.6]

            if high_similarity_matches:
                best_match = high_similarity_matches[0]
                title = best_match[0]
                metadata = best_match[1]
                similarity = best_match[2]
                logger.info(f"Found title match: '{title}' with similarity score: {similarity}")

                title_words = set(title.lower().split())
                query_words = query.lower().split()
                # Calculate word overlap to identify approximate position
                overlap_positions = []
                for i in range(len(query_words)):
                    if query_words[i] in title_words:
                        overlap_positions.append(i)

                if overlap_positions:
                    start_idx = min(overlap_positions)
                    end_idx = max(overlap_positions) + 1
                    clean_query_parts = query_words[:start_idx] + query_words[end_idx:]
                    clean_query = " ".join(clean_query_parts)

                    result = (title, clean_query)
                    if self.use_memory_cache:
                        self._query_cache[cache_key] = result
                    return result
                else:
                    result = (title, query)
                    if self.use_memory_cache:
                        self._query_cache[cache_key] = result
                    return result

        result = (None, query)
        if self.use_memory_cache:
            self._query_cache[cache_key] = result
        return result

    def _extract_title_with_gemini(self, query):
        """
        Use Gemini to extract movie title from a query
        """

        title_prompt = ChatPromptTemplate.from_template("""
        Extract the exact movie or tv series title from this query.
        Return ONLY the movie title with no additional text or explanations.
        If you cannot identify a movie title with certainty, return 'None'.

        Query: "{query}"

        Movie title:
        """)

        chain = LLMChain(llm=self.gemini, prompt=title_prompt)
        response = chain.invoke({"query": query})
        extracted_title = response['text'].strip()

        if extracted_title.lower() == 'none':
            return None

        if len(extracted_title.split()) <= 1 and len(extracted_title) <= 4:
            # Single very short word is likely not a movie title
            return None

        return extracted_title

    def answer_query(self, query):
        """
        Answer a query about a movie using detailed, structured context with LLM.
        Now leveraging both plot and Wikipedia introduction data directly.
        """

        title, actual_query = self.extract_title_from_query(query)
        if not title:
            # If no title was found, try a direct semantic search on the full query
            matches = self.find_best_title_match(query, top_k=3)
            logger.info(f"Semantic matches for '{query}': {matches}")  

            if matches and matches[0][2] > 0.75: 
                title = matches[0][0]
                metadata = matches[0][1]
                logger.info(f"No explicit title found, but semantic search suggests: {title}")
                actual_query = query
            else:
                return "I couldn't identify a movie in your question. Please ensure you include the movie name or specific details about the plot."

        # Get movie data - now retrieving plot, wiki_intro, and imdb_id all at once
        plot, wiki_intro, imdb_id = self.get_movie_data_by_title(title)
        if not plot and not wiki_intro:
            return f"I couldn't find information about '{title}'. Please check the movie title."

        additional_info = None
        if imdb_id:
            additional_info = self.get_movie_info_by_imdb_id(imdb_id)

        # Build a comprehensive context 
        context_lines = [
            f"Movie: {title}",
        ]

        # Add plot to context
        if plot and isinstance(plot, str) and plot.strip():
            context_lines.append(f"Plot: {plot}")

        # Add Wikipedia introduction to context
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

        # Check if we have sufficient information to answer
        if len(context_lines) < 3: 
            return f"I don't have enough information about '{title}' to answer your question properly."

        meta_prompt = ChatPromptTemplate.from_template("""
            You are an intelligent movie information system. You need to analyze the user's question and provide a tailored response about the movie.

            CONTEXT INFORMATION:
            {context}

            USER QUESTION: {user_question}

            INSTRUCTIONS:
            1. First, analyze what type of information the user is asking about (plot details, character information, factual data, opinion, etc.)
            2. If asked about the story, plot, narrative, premise, synopsis, storyline, or how the movie unfolds then retell the plot of '{movie_title}' as an engaging story for someone who hasn't seen it
            3. Use ALL available information - plot summary, Wikipedia introduction, and other details to craft a comprehensive answer
            4. If the Wikipedia introduction contains relevant information that's not in the plot, be sure to incorporate that as well
            5. Naturally weave in details like genre, main cast, director, release date, and other interesting facts from the context
            6. Craft a response that directly addresses their specific question
            7. If they're asking for facts or metadata, provide those specific details

            YOUR RESPONSE:
            """)

        logger.info(f"Constructed meta-prompt for LLM about {title} with enhanced context")

        try:
            chain = LLMChain(llm=self.groq, prompt=meta_prompt)
            response = chain.invoke({
                "context": context,
                "user_question": actual_query if actual_query else query,
                "movie_title": title
            })
            return response['text'].strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while trying to answer your question: {str(e)}"

    def build_index(self):
        """
        Process all content data, create embeddings, and store in Chroma DB.
        """
        if self.plots_df is not None:
            self._process_plots_batch()

        self._save_metadata()

        logger.info(f"Index built with {self.collection.count()} items.")

    def reset_index(self):
        """
        Delete and recreate the vector collection.
        """
        try:
            self.chroma_client.delete_collection("movie_titles")
            logger.info("Deleted existing collection.")

            self.collection = self.chroma_client.create_collection(
                name="movie_titles",
                embedding_function=self.embedding_function
            )
            logger.info("Created new collection.")
            if self.plots_df is not None:
                self.build_index()
            return True
        except Exception as e:
            logger.error(f"Error resetting index: {e}")
            return False


    def load_metadata(self, path="movie_qa.pkl"):
        """
        Load previously saved metadata
        """
        try:
            if os.path.exists(path):
                logger.info(f"Loading metadata from {path}")
                with open(path, 'rb') as f:
                    metadata = pickle.load(f)

                self.title_to_imdb = metadata.get('title_to_imdb', {})
                logger.info("Metadata loaded successfully")
            else:
                logger.warning(f"Metadata file {path} not found")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise


plots_path = "/content/drive/MyDrive/plots.csv"
additional_data_path = "/content/drive/MyDrive/content_descriptions.pkl"
chroma_persist_directory = "/content/drive/MyDrive/chroma_db/plots"
metadata_cache_path = "/content/drive/MyDrive/movie_qa.pkl"

os.makedirs(chroma_persist_directory, exist_ok=True)
os.makedirs(os.path.dirname(metadata_cache_path), exist_ok=True)

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
try:
    qa_system = MovieQASystem(
        plots_path=plots_path,
        additional_data_path=additional_data_path,
        gemini_api_key=GEMINI_API_KEY,
        groq_api_key=GROQ_API_KEY,
        chroma_persist_directory=chroma_persist_directory,
        metadata_cache_path=metadata_cache_path
    )

    qa_system.load_metadata(metadata_cache_path)
    try:
        title_collection = qa_system.chroma_client.get_collection("movie_titles")
        if title_collection.count() == 0:
            logger.info("ChromaDB collection exists but is empty. Loading data from CSV...")
            qa_system.load_csv_data(plots_path)
            qa_system.save_metadata(metadata_cache_path)
    except Exception as e:
        logger.info(f"Creating new ChromaDB collection: {e}")
        qa_system.load_csv_data(plots_path)
        qa_system.save_metadata(metadata_cache_path)

    if os.path.exists(additional_data_path):
        qa_system.load_pickle_data(additional_data_path)

except Exception as e:
    logger.error(f"Failed to initialize QA system: {e}")
    raise


qq= ["what is the main story of dune part two",
        "In the movie 'The Shawshank Redemption', who is the main character?",
        "What happens at the end of Inception?",
        "Tell me about the plot of 'The Godfather'",
        "Who played the main villain in Dark Knight?",
        "What's the twist in 'Fight Club'?"]
for query in qq:
    answer = qa_system.answer_query(query)
    print(f"Q: {query}")
    print(f"A: {answer}")