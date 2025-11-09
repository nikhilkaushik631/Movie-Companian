"""
Dedicated AI Summary Generator for Movie/TV Titles

This module creates AI summaries using TMDB plot data and a dedicated LLM chat,
separate from the main recommendation and QA systems.
"""

import os
import httpx
import asyncio
from typing import Optional, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SummaryLLMChat:
    """
    Dedicated LLM chat specifically for generating movie/TV summaries
    """

    def __init__(self, groq_api_key: str = None):
        # Try multiple environment variable names and load .env files
        load_dotenv()
        load_dotenv('env')

        self.groq_api_key = (
            groq_api_key or
            os.getenv("GROQ_API_KEY") or
            os.getenv("groq_api_key")
        )

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for summary generation. Check your .env file.")

        self.client = Groq(api_key=self.groq_api_key)
        self.model = "llama-3.1-8b-instant"  # Updated fast model for summaries

    async def generate_summary(self, prompt: str) -> str:
        """Generate summary using Groq LLM"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional movie critic and entertainment writer. Generate engaging, informative summaries for movie and TV shows. IMPORTANT: Always provide a complete summary based on the available plot information. Never ask questions or say you need more information. Never use phrases like 'What would you like to know?' or 'What specific aspect?'. Always write a complete summary covering the plot, themes, and what makes the title worth watching. Write in a descriptive, engaging tone."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                max_tokens=250,
                temperature=0.6,
                top_p=0.9,
            )

            summary = chat_completion.choices[0].message.content.strip()

            # Validate the response to avoid questions
            invalid_patterns = [
                "what would you like",
                "what specific aspect",
                "which aspect",
                "what do you want to know",
                "?",  # Questions
                "more information needed"
            ]

            # Check if response contains invalid patterns
            summary_lower = summary.lower()
            if any(pattern in summary_lower for pattern in invalid_patterns):
                # Regenerate with more explicit instructions
                return None

            return summary

        except Exception as e:
            print(f"Error generating summary: {e}")
            # Return None instead of error message so it can be filtered out
            return None


class TMDBDataFetcher:
    """
    Fetches comprehensive data from TMDB for summary generation
    """

    def __init__(self, api_key: str = None, bearer_token: str = None):
        # Load environment variables
        load_dotenv()
        load_dotenv('env')

        self.api_key = (
            api_key or
            os.getenv("TMDB_API_KEY") or
            os.getenv("tmdb_api_key")
        )
        self.bearer_token = (
            bearer_token or
            os.getenv("TMDB_BEARER_TOKEN") or
            os.getenv("tmdb_bearer_token")
        )
        self.base_url = "https://api.themoviedb.org/3"

        if not self.api_key and not self.bearer_token:
            raise ValueError("TMDB_API_KEY or TMDB_BEARER_TOKEN is required")

    def _get_headers(self):
        """Get request headers for TMDB API"""
        headers = {"accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _get_params(self):
        """Get request parameters for TMDB API"""
        params = {}
        if self.api_key and not self.bearer_token:
            params["api_key"] = self.api_key
        return params

    async def fetch_movie_details(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed movie information including plot"""
        try:
            url = f"{self.base_url}/movie/{movie_id}"
            params = self._get_params()
            headers = self._get_headers()

            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            print(f"Error fetching movie details: {e}")

        return None

    async def fetch_tv_details(self, tv_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed TV show information including plot"""
        try:
            url = f"{self.base_url}/tv/{tv_id}"
            params = self._get_params()
            headers = self._get_headers()

            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            print(f"Error fetching TV details: {e}")

        return None

    async def fetch_cast_and_crew(self, content_id: str, content_type: str = "movie") -> Optional[Dict[str, Any]]:
        """Fetch cast and crew information"""
        try:
            url = f"{self.base_url}/{content_type}/{content_id}/credits"
            params = self._get_params()
            headers = self._get_headers()

            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            print(f"Error fetching credits: {e}")

        return None


class DedicatedSummaryGenerator:
    """
    Main class that combines TMDB data fetching with LLM summary generation
    """

    def __init__(self):
        self.llm_chat = SummaryLLMChat()
        self.tmdb_fetcher = TMDBDataFetcher()

    async def generate_comprehensive_summary(
        self,
        external_id: str,
        title: str,
        media_type: str = "movie",
        summary_type: str = "card"
    ) -> str:
        """
        Generate comprehensive AI summary using TMDB plot data

        Args:
            external_id: TMDB ID
            title: Title name
            media_type: 'movie' or 'tv'
            summary_type: 'card' (short) or 'detailed' (long)
        """
        try:
            # Fetch detailed information from TMDB
            if media_type == "tv":
                details = await self.tmdb_fetcher.fetch_tv_details(external_id)
            else:
                details = await self.tmdb_fetcher.fetch_movie_details(external_id)

            if not details:
                return f"A compelling {media_type} worth exploring."

            # Fetch cast and crew
            credits = await self.tmdb_fetcher.fetch_cast_and_crew(external_id, media_type)

            # Extract key information
            plot = details.get('overview', '')
            genres = [g['name'] for g in details.get('genres', [])[:3]]  # Top 3 genres
            rating = details.get('vote_average', 0)
            year = None

            if media_type == "movie":
                release_date = details.get('release_date', '')
                year = release_date.split('-')[0] if release_date else ''
                runtime = details.get('runtime', 0)
            else:
                first_air_date = details.get('first_air_date', '')
                year = first_air_date.split('-')[0] if first_air_date else ''
                runtime = details.get('episode_run_time', [0])[0] if details.get('episode_run_time') else 0

            # Extract cast information
            main_cast = []
            if credits and 'cast' in credits:
                main_cast = [actor['name'] for actor in credits['cast'][:4]]  # Top 4 actors

            # Extract director/creator information
            key_crew = []
            if credits and 'crew' in credits:
                for person in credits['crew']:
                    if person['job'] in ['Director', 'Creator', 'Executive Producer']:
                        key_crew.append(f"{person['name']} ({person['job']})")
                        if len(key_crew) >= 2:  # Limit to 2 key crew members
                            break

            # Create comprehensive prompt with all available data
            type_label = "TV series" if media_type == "tv" else "film"
            year_part = f" ({year})" if year else ""
            genre_text = f" ({', '.join(genres)})" if genres else ""

            if summary_type == "card":
                # Short summary for cards (2-3 sentences)
                prompt = f"""Write a compelling 2-3 sentence summary for the {type_label} "{title}"{year_part}{genre_text} for a recommendation card.

Plot: {plot}

Key Cast: {', '.join(main_cast) if main_cast else 'Not available'}
Key Crew: {', '.join(key_crew) if key_crew else 'Not available'}
Rating: {rating}/10
{"Runtime: " + str(runtime) + " minutes" if runtime else ""}

Focus on what makes this {type_label} engaging and worth watching. Keep it under 150 words and make it exciting."""

            else:
                # Detailed summary for modals
                prompt = f"""Based on the plot information provided, write a comprehensive 4-5 sentence summary for the {type_label} "{title}"{year_part}{genre_text}.

Plot: {plot}

Key Cast: {', '.join(main_cast) if main_cast else 'Not available'}
Key Crew: {', '.join(key_crew) if key_crew else 'Not available'}
Rating: {rating}/10
{"Runtime: " + str(runtime) + " minutes" if runtime else ""}

IMPORTANT: Write a complete, informative summary NOW based on the plot information above. Do NOT ask questions. Include:
- Plot highlights and key story elements
- What makes this {type_label} compelling
- The tone and themes explored
- Why viewers should watch it

Write in an engaging, descriptive style. Start with the plot overview and flow naturally."""

            # Generate summary using dedicated LLM
            summary = await self.llm_chat.generate_summary(prompt)

            # If LLM generation failed, return None so it can be filtered out
            if summary is None:
                return None

            return summary

        except Exception as e:
            print(f"Error generating comprehensive summary: {e}")
            return None

    async def generate_bulk_summaries(self, titles_data: list, summary_type: str = "card") -> Dict[str, str]:
        """
        Generate summaries for multiple titles

        Args:
            titles_data: List of dicts with keys: external_id, title, media_type
            summary_type: 'card' or 'detailed'

        Returns:
            Dict mapping external_id to generated summary
        """
        results = {}

        for item in titles_data:
            external_id = item.get('external_id')
            title = item.get('title', '')
            media_type = item.get('media_type', 'movie')

            if not external_id or not title:
                continue

            try:
                summary = await self.generate_comprehensive_summary(
                    external_id, title, media_type, summary_type
                )
                results[external_id] = summary

                # Small delay to respect rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"Error generating summary for {title}: {e}")
                results[external_id] = f"An engaging {media_type} worth exploring."

        return results


# Global instance for use in main.py
summary_generator = None

def get_summary_generator():
    """Get or create the global summary generator instance"""
    global summary_generator
    if summary_generator is None:
        try:
            summary_generator = DedicatedSummaryGenerator()
        except Exception as e:
            print(f"Error initializing summary generator: {e}")
            summary_generator = None
    return summary_generator


# Async function for use in FastAPI endpoints
async def generate_ai_summary(external_id: str, title: str, media_type: str = "movie", summary_type: str = "card") -> str:
    """
    Generate AI summary using dedicated LLM and TMDB data

    This is the main function to call from FastAPI endpoints
    """
    generator = get_summary_generator()
    if not generator:
        return None

    summary = await generator.generate_comprehensive_summary(external_id, title, media_type, summary_type)
    return summary


if __name__ == "__main__":
    # Test the summary generator
    async def test():
        generator = DedicatedSummaryGenerator()

        # Test with a popular movie
        summary = await generator.generate_comprehensive_summary(
            external_id="550",  # Fight Club
            title="Fight Club",
            media_type="movie",
            summary_type="card"
        )

        print("Generated Summary:")
        print(summary)

    asyncio.run(test())