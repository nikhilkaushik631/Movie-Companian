# RAG Content Recommender System using Chroma
# Part 1: Architecture and Dependencies

import json
from typing import List, Dict
import os
import warnings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

class RAGContentRecommender:
    """
    A recommendation system that combines vector similarity search with LLM-enhanced output generation.
    Uses Chroma as the vector database instead of FAISS.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MPNet-base-v2",
        groq_api_key=os.environ.get("groq_api_key"),
        pinecone_api_key=os.environ.get("pinecone_api_key"),
        pinecone_index_name: str = "content-collection",
        use_memory_cache: bool = True,
        descriptions_cache_path=os.environ.get("description_path")
    ):
        # Set up embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.use_memory_cache = use_memory_cache
        self.descriptions_cache_path = descriptions_cache_path

        if not pinecone_api_key:
            raise ValueError("Pinecone API key is required.")
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name

        # Set up Pinecone client
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            if self.pinecone_index_name not in self.pc.list_indexes().names():
                 raise RuntimeError(f"Pinecone index '{self.pinecone_index_name}' does not exist. "
                                    "Please create and populate it before using this system.")
            self.index = self.pc.Index(self.pinecone_index_name)
            print(f"Successfully connected to Pinecone index '{self.pinecone_index_name}'.")
            print("Index Stats: ",self.index.describe_index_stats()) # Print stats for confirmation

        except Exception as e:
            raise RuntimeError(f"Could not initialize Pinecone: {e}") from e


        self.content_descriptions = {}
        self._load_content_descriptions()

        self.groq2 = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )



    def _load_content_descriptions(self):
        """
        Load content descriptions from disk if available.
        """
        if os.path.exists(self.descriptions_cache_path):
            try:
                with open(self.descriptions_cache_path, 'rb') as f:
                    self.content_descriptions = pickle.load(f)
                return True
            except Exception as e:
                print(f"Error loading content descriptions: {e}")
                self.content_descriptions = {}
                return False
        else:
            print("No saved content descriptions found.")
            self.content_descriptions = {}
            return False


    async def _analyze_query_context(self, query: str) -> Dict:
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
            # loop = asyncio.get_event_loop()
            # response = await loop.run_in_executor(None, chain.invoke, {"query": query})
            response = await chain.ainvoke({"query": query})

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
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Enhanced search with simplified scoring and content type filtering.
        """
        query_lower = query.lower()
        query_context = await self._analyze_query_context(query_lower)

        search_k = min(top_k * 5, 20)

        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                query_vector = await loop.run_in_executor(
                    pool, 
                    self.embedding_model.encode,
                    query
                )
            query_vector = query_vector.tolist()

            results = await asyncio.to_thread(
                self.index.query,
                vector=query_vector,
                top_k=search_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error during Pinecone query: {e}")
            return []

        if not results or not results.get('matches'):
            return []

        all_items = []
        for match in results['matches']:
            content_id = match.get('id')
            metadata = match.get('metadata', {})
            score = match.get('score', 0.0)
            description = self.content_descriptions.get(content_id)

            try:
                content_data = json.loads(description)
                basic_info = content_data.get('basic_info', {})
                content_type = basic_info.get('type', '')
                title = basic_info.get('title', metadata.get('title', 'Unknown'))

                if query_context['content_type'] != 'both':
                    if (query_context['content_type'] == 'movie' and content_type != 'Movie') or \
                       (query_context['content_type'] == 'tv' and content_type != 'TV Show'):
                        continue


                all_items.append({
                    'id': content_id,
                    'title': title,
                    'type': content_type,
                    'score': score,
                    'description': description
                })

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Skipping item {content_id} due to data error: {e}")
                continue

        all_items.sort(key=lambda x: x['score'], reverse=True)
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
        - When the user asks for recent movies/shows, ONLY consider content from 2021-2025
        - Do not recommend content same as the title in the query. limit the recommendation to top 3.
        - Do not reveal raw CSV fields. Avoid mentioning terms like release date, popularity, votes, or ratings.
        - If JSON data is not available, rely on embedding-based insights.
        - Ensure recommendations are limited to content of type "{content_type}".
        - For queries about directors, only include recommendations by the mentioned director.
        - if asked about ratings then you can mention it in the description. if the rating is above 8/10 then it is highly rated
        - Highlight unique aspects of each recommendation rather than generic descriptions.
        - Reasons should be concise and relevant to the query.
        - If the recommender cannot find any recommendations: first try to answer it from your memory, if can't then ask user to ask different or modify the query

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

    async def recommend(self, query: str, top_k: int = 3) -> str:
        """
        Generate recommendations based on a natural language query.
        """

        retrieved_content = await self.search(query, top_k=top_k*3)
        if not retrieved_content:
            return "No relevant content found. Please try a different query."

        prompt_text = self._create_prompt(query, retrieved_content)
        meta_prompt = ChatPromptTemplate.from_template("{prompt_text}")

        try:
            chain = LLMChain(llm=self.groq2, prompt=meta_prompt)
            # loop = asyncio.get_event_loop()
            # response = await loop.run_in_executor(None, chain.invoke, {"prompt_text": prompt_text})
            response = await chain.ainvoke({"prompt_text": prompt_text})
            return response['text'].strip()
        except Exception as e:
            return f"Error generating recommendations: {e}"





async def main():

    recommender = RAGContentRecommender()

    queries = [
        # "Shows like Breaking Bad with a complex protagonist",
        # "Movies directed by Christopher Nolan",
        "which are the top hbo shows",
        "top paramount movies",
        # "Feel-good comedies of male leads",
        # "Science fiction TV shows with time travel",
        # "Recent action movies with high ratings",
        # "movies with great cgi like dune part two",
        # "japanese anime like bleach",
        # "movies like interstellar",
        # "recommend best marvel movies",
        # "best batman movies",
        # "best dc movies"

    ]

    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")

        # Get recommendations
        recommendations = await recommender.recommend(query, top_k=5)

        # Print recommendations
        print(recommendations)
        print("\n" + "-"*80)

if __name__ == "__main__":
    asyncio.run(main())