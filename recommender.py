# RAG Content Recommender System using Chroma
# Part 1: Architecture and Dependencies

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Union, Tuple, Optional
import os
import math
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pickle

class RAGContentRecommender:
    """
    A recommendation system that combines vector similarity search with LLM-enhanced output generation.
    Uses Chroma as the vector database instead of FAISS.
    """

    def __init__(
        self,
        movies_path: str = "/content/drive/MyDrive/movies.csv",
        tv_shows_path: str = "/content/drive/MyDrive/tv_series.csv",
        embedding_model_name: str = "all-MPNet-base-v2",
        groq_api_key: str = None,
        chroma_persist_directory: str = "/content/drive/MyDrive/chroma_db/recommender",
        use_memory_cache: bool = True,
        descriptions_cache_path: str = "/content/drive/MyDrive/content_descriptions.pkl"
    ):

        # Set up embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.use_memory_cache = use_memory_cache
        self._query_cache = {}
        self.descriptions_cache_path = descriptions_cache_path

        # Set up Chroma client with explicit settings
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
            print(f"Warning: Failed to initialize ChromaDB with custom settings: {e}")
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)

        # Create a sentence transformer embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        self._initialize_collection()
        self.content_descriptions = {}
        self._load_content_descriptions()
        self.movies_df = None
        self.tv_shows_df = None

        if movies_path and os.path.exists(movies_path):
            self.movies_df = pd.read_csv(movies_path)
            self.movies_df['content_type'] = 'movie'
            print(f"Loaded {len(self.movies_df)} movies.")

        if tv_shows_path and os.path.exists(tv_shows_path):
            self.tv_shows_df = pd.read_csv(tv_shows_path)
            self.tv_shows_df['content_type'] = 'tv'
            print(f"Loaded {len(self.tv_shows_df)} TV shows.")

        self.groq2 = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )


    def _initialize_collection(self):
        """
        Initialize the ChromaDB collection with proper error handling.
        """
        collection_name = "content_collection"

        # First try to get the existing collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Successfully connected to existing collection '{collection_name}'.")
            return
        except Exception as e:
            print(f"Did not find existing collection: {e}")

        # If retrieving fails, try to create a new collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Successfully created new collection '{collection_name}'.")
            return
        except Exception as e:
            print(f"Failed to create new collection: {e}")

        # If both attempts fail, check if we have any collections at all
        try:
            collections = self.chroma_client.list_collections()
            if collections:
                print(f"Found existing collections: {[c.name for c in collections]}")

                # Try to reset collection as a last resort
                try:
                    print(f"Attempting to reset collection '{collection_name}'...")
                    self.chroma_client.delete_collection(collection_name)
                    self.collection = self.chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    print(f"Successfully reset and created collection '{collection_name}'.")
                    return
                except Exception as reset_error:
                    print(f"Failed to reset collection: {reset_error}")
            else:
                print("No collections found in the database.")
        except Exception as list_error:
            print(f"Failed to list collections: {list_error}")

        raise RuntimeError("Failed to initialize ChromaDB collection after multiple attempts. Check your database configuration.")

    # Part 2: Data Processing and Embedding Creation
    def _save_content_descriptions(self):
        """
        Save content descriptions to disk.
        """
        try:
            with open(self.descriptions_cache_path, 'wb') as f:
                pickle.dump(self.content_descriptions, f)
            print(f"Saved {len(self.content_descriptions)} content descriptions to {self.descriptions_cache_path}")
            return True
        except Exception as e:
            print(f"Error saving content descriptions: {e}")
            return False

    def _load_content_descriptions(self):
        """
        Load content descriptions from disk if available.
        """
        if os.path.exists(self.descriptions_cache_path):
            try:
                with open(self.descriptions_cache_path, 'rb') as f:
                    self.content_descriptions = pickle.load(f)
                print(f"Loaded {len(self.content_descriptions)} content descriptions from {self.descriptions_cache_path}")
                return True
            except Exception as e:
                print(f"Error loading content descriptions: {e}")
                self.content_descriptions = {}
                return False
        else:
            print("No saved content descriptions found.")
            self.content_descriptions = {}
            return False

    def build_index(self):
        """
        Process all content data, create embeddings, and store in Chroma DB.
        """

        if self.movies_df is not None:
            self._process_movies_batch()

        if self.tv_shows_df is not None:
            self._process_tv_shows_batch()

        self._save_content_descriptions()

        print(f"Index built with {len(self.content_descriptions)} items.")

    def _process_movies_batch(self):
        """
        Process all movies in batches for better performance.
        """
        print("Processing movies in batches...")

        BATCH_SIZE = 100 

        total_movies = len(self.movies_df)
        batch_count = (total_movies + BATCH_SIZE - 1) // BATCH_SIZE 

        for batch_idx in range(batch_count):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_movies)

            print(f"Processing movie batch {batch_idx+1}/{batch_count} (items {start_idx}-{end_idx})")
            batch_df = self.movies_df.iloc[start_idx:end_idx]

            batch_ids = []
            batch_texts = []
            batch_metadatas = []

            for idx, row in batch_df.iterrows():
                try:
                    # Create a unique ID for the movie
                    if 'imdb_id' in row and pd.notna(row['imdb_id']):
                        content_id = f"movie_{row['imdb_id']}"
                    else:
                        title_part = row.get('title', '').replace(' ', '_')[:20] if pd.notna(row.get('title')) else ''
                        content_id = f"movie_{idx}_{title_part}"

                    # Create combined feature text for embedding
                    features_text = self._combine_movie_features(row)
                    content_description = self._create_content_description(row)
                    # Store description for later use
                    self.content_descriptions[content_id] = content_description

                    # Add to batch
                    batch_ids.append(content_id)
                    batch_texts.append(features_text)
                    batch_metadatas.append({
                        "title": row.get('title', 'Unknown'),
                        "type": "movie",
                        "rating": float(row.get('vote_average', 0) or 0),
                        "popularity": float(row.get('popularity', 0) or 0)
                    })

                except Exception as e:
                    print(f"Error processing movie {row.get('title', 'unknown')}: {e}")

            # Add batch to collection
            if batch_ids:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )

    def _process_tv_shows_batch(self):
        """
        Process all TV shows in batches for better performance.
        """
        print("Processing TV shows in batches...")

        BATCH_SIZE = 100 

        total_shows = len(self.tv_shows_df)
        batch_count = (total_shows + BATCH_SIZE - 1) // BATCH_SIZE  

        for batch_idx in range(batch_count):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_shows)

            print(f"Processing TV show batch {batch_idx+1}/{batch_count} (items {start_idx}-{end_idx})")
            batch_df = self.tv_shows_df.iloc[start_idx:end_idx]

            batch_ids = []
            batch_texts = []
            batch_metadatas = []

            for idx, row in batch_df.iterrows():
                try:
                    # Create a unique ID for the TV show
                    if 'imdb_id' in row and pd.notna(row['imdb_id']):
                        content_id = f"tv_{row['imdb_id']}"
                    else:
                        title_part = row.get('title', '').replace(' ', '_')[:20] if pd.notna(row.get('title')) else ''
                        content_id = f"tv_{idx}_{title_part}"

                    # Create combined feature text for embedding
                    features_text = self._combine_tv_features(row)
                    content_description = self._create_content_description(row)

                    # Store description for later use
                    self.content_descriptions[content_id] = content_description

                    # Add to batch
                    batch_ids.append(content_id)
                    batch_texts.append(features_text)
                    batch_metadatas.append({
                        "title": row.get('title', 'Unknown'),
                        "type": "tv",
                        "rating": float(row.get('vote_average', 0) or 0),
                        "popularity": float(row.get('popularity', 0) or 0)
                    })

                except Exception as e:
                    print(f"Error processing TV show {row.get('title', 'unknown')}: {e}")

            # Add batch to collection
            if batch_ids:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )

    def _combine_movie_features(self, row):
        """
        Combine movie features into a single text representation for embedding.
        """
        features = []

        if 'title' in row and pd.notna(row['title']):
            features.append(f"Title: {row['title']}")
        if 'overview' in row and pd.notna(row['overview']):
            features.append(f"Overview: {row['overview']}")
        if 'genres' in row and pd.notna(row['genres']):
            features.append(f"Genres: {row['genres']}")
        if 'keywords' in row and pd.notna(row['keywords']):
            features.append(f"Keywords: {row['keywords']}")
        if 'cast' in row and pd.notna(row['cast']):
            features.append(f"Cast: {row['cast']}")
        if 'directors' in row and pd.notna(row['directors']):
            features.append(f"Directors: {row['directors']}")
        if 'budget' in row and pd.notna(row['budget']):
            features.append(f"Budget: {row['budget']}")
        if 'collection' in row and pd.notna(row['collection']):
            features.append(f"Collection: {row['collection']}")
        if 'original_language' in row and pd.notna(row['original_language']):
            features.append(f"Original Language: {row['original_language']}")
        if 'country_of_origin' in row and pd.notna(row['country_of_origin']):
            features.append(f"Country of Origin: {row['country_of_origin']}")
        if 'popularity' in row and pd.notna(row['popularity']):
            features.append(f"Popularity: {row['popularity']}")
        if 'vote_count' in row and pd.notna(row['vote_count']):
            features.append(f"Vote Count: {row['vote_count']}")
        if 'rating' in row and pd.notna(row['rating']):
            features.append(f"Rating: {row['rating']}")
        if 'production_details' in row and pd.notna(row['production_details']):
            features.append(f"Production Details: {row['production_details']}")
        if 'runtime' in row and pd.notna(row['runtime']):
            features.append(f"Runtime: {row['runtime']}")
        if 'release_date' in row and pd.notna(row['release_date']):
            features.append(f"Release Date: {row['release_date']}")

        return " ".join(features)

    def _combine_tv_features(self, row):
        """
        Combine TV show features into a single text representation for embedding.
        """
        features = []

        # Add title
        if 'title' in row and pd.notna(row['title']):
            features.append(f"Title: {row['title']}")
        if 'overview' in row and pd.notna(row['overview']):
            features.append(f"Overview: {row['overview']}")
        if 'genres' in row and pd.notna(row['genres']):
            features.append(f"Genres: {row['genres']}")
        if 'keywords' in row and pd.notna(row['keywords']):
            features.append(f"Keywords: {row['keywords']}")
        if 'cast' in row and pd.notna(row['cast']):
            features.append(f"Cast: {row['cast']}")
        if 'seasons_count' in row and pd.notna(row['seasons_count']):
            features.append(f"Seasons: {row['seasons_count']}")
        if 'episodes_count' in row and pd.notna(row['episodes_count']):
            features.append(f"Episodes: {row['episodes_count']}")
        if 'air_years' in row and pd.notna(row['air_years']):
            features.append(f"Air Years: {row['air_years']}")
        if 'country_of_origin' in row and pd.notna(row['country_of_origin']):
            features.append(f"Country of Origin: {row['country_of_origin']}")
        if 'popularity' in row and pd.notna(row['popularity']):
            features.append(f"Popularity: {row['popularity']}")
        if 'vote_count' in row and pd.notna(row['vote_count']):
            features.append(f"Vote Count: {row['vote_count']}")
        if 'rating' in row and pd.notna(row['rating']):
            features.append(f"Rating: {row['rating']}")
        if 'production_details' in row and pd.notna(row['production_details']):
            features.append(f"Production Details: {row['production_details']}")
        if 'original_language' in row and pd.notna(row['original_language']):
            features.append(f"Original Language: {row['original_language']}")
        if 'networks' in row and pd.notna(row['networks']):
            features.append(f"Networks: {row['networks']}")

        return " ".join(features)

    # Part 3: Content Description Creation with JSON Structure
    def _create_content_description(self, row):
        """Create structured JSON representation of content with all available features."""
        # Basic information available for all content
        content_dict = {
            "basic_info": {
                "title": row.get('title', None),
                "type": 'Movie' if row['content_type'] == 'movie' else 'TV Show',
                "rating": row.get('rating', None),
                "popularity": row.get('popularity', None),
                "vote_count": row.get('vote_count', None),
                "original_language": row.get('original_language', None),
                "country_of_origin": row.get('country_of_origin', None)
            },
            "content": {
                "overview": row.get('overview', None),
                "genres": row.get('genres', None),
                "keywords": row.get('keywords', None),
                "production_details": row.get('production_details', None)
            },
            "people": {
                "cast": row.get('cast', None)
            }
        }

        if row['content_type'] == 'movie':
            content_dict["movie_specific"] = {
                "release_date": row.get('release_date', None),
                "directors": row.get('directors', None),
                "budget": row.get('budget', None),
                "collection": row.get('collection', None),
                "runtime": row.get('runtime', None)
            }

        else:
            content_dict["tv_specific"] = {
                "seasons_count": row.get('seasons_count', None),
                "episodes_count": row.get('episodes_count', None),
                "air_years": row.get('air_years', None),
                "networks": row.get('networks', None)
            }

        # Clean up None values for better JSON representation
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v is not None}
            return d

        content_dict = clean_dict(content_dict)

        # Convert to formatted JSON string
        return json.dumps(content_dict, indent=2)


    def _analyze_query_context(self, query: str) -> Dict:
        """
        Use LLM to analyze query context and extract structured information.
        """
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

        Respond ONLY with valid JSON in this exact format without any backticks, markdown formatting, or explanatory text:
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
            chain = LLMChain(llm=self.groq2, prompt=template)

            response = chain.invoke({"query": query})

            raw_text = response['text'].strip()

            if '```' in raw_text:
                code_block_parts = raw_text.split('```')
                if len(code_block_parts) >= 3: 
                    raw_text = code_block_parts[1]
                    # Remove language specifier if present
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
                print(f"Attempted to parse: {raw_text}")
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


    # Part 4: Search and Recommendation     
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Enhanced search with simplified scoring and content type filtering.
        """
        query_lower = query.lower()
        query_context = self._analyze_query_context(query_lower)

        search_k = min(top_k * 5, 20)

        # Query the Chroma collection
        results = self.collection.query(
            query_texts=[query],
            n_results=search_k,
            include=["metadatas", "distances", "documents"]
        )

        if not results or 'ids' not in results or len(results['ids']) == 0:
            return []

        all_items = []
        for i, content_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            description = self.content_descriptions.get(content_id, "{}")

            try:
                content_data = json.loads(description)
                content_type = content_data.get('basic_info', {}).get('type', '')
                
                if query_context['content_type'] != 'both':
                    if (query_context['content_type'] == 'movie' and content_type != 'Movie') or \
                    (query_context['content_type'] == 'tv' and content_type != 'TV Show'):
                        continue

                # Simple scoring based on semantic similarity
                score = distance

                # Apply small bonus for exact content type match
                if query_context['content_type'] != 'both' and \
                ((query_context['content_type'] == 'movie' and content_type == 'Movie') or \
                    (query_context['content_type'] == 'tv' and content_type == 'TV Show')):
                    score *= 0.9

                all_items.append({
                    'id': content_id,
                    'title': metadata.get('title', 'Unknown'),
                    'type': content_type,
                    'score': score,
                    'description': description
                })

            except (json.JSONDecodeError, ValueError) as e:
                continue

        # Sort by score 
        all_items.sort(key=lambda x: x['score'])
        return all_items[:top_k]


    # Part 5: Prompt Engineering for LLM
    def _create_prompt(self, query: str, retrieved_content: List[Dict]) -> str:
        """
        Create a concise prompt that instructs the LLM to recommend content
        strictly matching the extracted context.
        """

        query_lower = query.lower()
        if any(word in query_lower for word in ['movie', 'film']):
            content_type = "movies"
        elif any(word in query_lower for word in ['tv', 'show', 'series']):
            content_type = "tv series"
        else:
            content_type = "movies and tv series"

        prompt = f""""You are an expert media recommendation system.
        Based on the user's query and the JSON content information I recommend the most relevant content.
        For the query: {query}, Recommend the best {content_type} strictly matching the following context.
        - Do not reveal raw CSV fields. Avoid mentioning terms like release date, popularity, votes, or ratings.
        - If JSON data is not available, rely on embedding-based insights.
        - Ensure recommendations are limited to content of type "{content_type}".
        - For queries about directors, only include recommendations by the mentioned director.
        - For queries indicating 'recent' content, consider only items from 2021 to 2025.
        - Include rating in the reasoning only if it is more than 7 and note it as high if above 8.
        - Highlight unique aspects of each recommendation rather than generic descriptions.
        - Reasons should be concise and relevant to the query.
        - If the recommender cannot find any recommendations: first try to answer it from your memory, if cant then ask user to ask different or modify the query

        Retrieved Content Information (in JSON format):
        """
        for i, content in enumerate(retrieved_content, 1):
            prompt += f"\n--- Item {i} ---\n{content['description']}\n"

        prompt += """
        Format your response as follows (response should not include any text other than the format):

        ## Recommendations
        1. [Title] ([Movie/TV Show], [Year]) - [a brief reason for recommendation]
        2. [Title] ([Movie/TV Show], [Year]) - [a brief reason for recommendation]
        3. [Title] ([Movie/TV Show], [Year]) - [a brief reason for recommendation]

        """
        return prompt


    def recommend(self, query: str, top_k: int = 8) -> str:
        """
        Generate recommendations based on a natural language query.
        """

        retrieved_content = self.search(query, top_k)
        if not retrieved_content:
            return "No relevant content found. Please try a different query."

        prompt_text = self._create_prompt(query, retrieved_content)
        meta_prompt = ChatPromptTemplate.from_template("{prompt_text}")

        try:
            chain = LLMChain(llm=self.groq2, prompt=meta_prompt)
            response = chain.invoke({"prompt_text": prompt_text})
            return response['text'].strip()
        except Exception as e:
            return f"Error generating recommendations: {e}"


    # Part 6: Utility Methods & Complete Class
    def get_collection_stats(self):
        """
        Get statistics about the vector collection.
        """
        try:
            count = self.collection.count()
            return {
                "total_items": count,
                "movies": len([id for id in self.content_descriptions if id.startswith("movie_")]) if self.content_descriptions else "Unknown",
                "tv_shows": len([id for id in self.content_descriptions if id.startswith("tv_")]) if self.content_descriptions else "Unknown",
            }
        except Exception as e:
            return {"error": str(e)}

    def reset_index(self):
        """
        Delete and recreate the vector collection.
        """
        try:
            # Delete collection
            self.chroma_client.delete_collection("content_collection")
            print("Deleted existing collection.")

            # Recreate collection
            self.collection = self.chroma_client.create_collection(
                name="content_collection",
                embedding_function=self.embedding_function
            )
            print("Created new collection.")

            # Clear content descriptions
            self.content_descriptions = {}

            return True
        except Exception as e:
            print(f"Error resetting index: {e}")
            return False



# Initialize the recommender
recommender = RAGContentRecommender(
    movies_path="/content/drive/MyDrive/movies.csv",
    tv_shows_path="/content/drive/MyDrive/tv_series.csv",
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    chroma_persist_directory="/content/drive/MyDrive/chroma_db",
    descriptions_cache_path="/content/drive/MyDrive/content_descriptions.pkl"
)

# Check if collection has items and build index if needed
stats = recommender.get_collection_stats()
if stats.get("total_items", 0) == 0:
    print("Building index...")
    recommender.build_index()
else:
    print(f"Using existing index with {stats.get('total_items')} items.")
queries = [
    "Shows like Breaking Bad with a complex protagonist",
    "Movies directed by Christopher Nolan",
    "Feel-good comedies with strong female leads",
    "Science fiction TV shows with time travel",
    "Recent action movies with high ratings",
    "movies with great cgi like dune part two",
    "japanese anime like bleach",
    "movies like interstellar",
    "recommend best marvel movies",
    "best batman movies",
    "best dc movies"

]

# Process each query
for i, query in enumerate(queries, 1):
    print(f"\n--- Query {i}: {query} ---")

    # Get recommendations
    recommendations = recommender.recommend(query, top_k=5)

    # Print recommendations
    print(recommendations)
    print("\n" + "-"*80)
