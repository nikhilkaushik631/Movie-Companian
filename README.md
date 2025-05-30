# Entertainment Companion Chatbot- using RAG and AI Agents

A powerful entertainment assistant that provides movie/TV recommendations and answers questions about entertainment content using Retrieval Augmented Generation (RAG) and AI agents.

## Project Overview

This project consists of two main components:
1. **Recommendation System** - Provides personalized movie and TV show recommendations based on user queries
2. **Q&A System** - Answers questions about movies, TV shows, and entertainment topics

## Architecture

### Recommendation System

**Data Collection & Processing:**
* **Data Source**: Scraped IMDb for 10,000+ movies and TV shows (worldwide and Indian content)
* **Dataset Structure**: Two main CSV files - one with comprehensive title information, another with detailed plots
* **Embedding Generation**: Used sentence transformers - all-MPNET-base-v2 to create embeddings of content descriptions
* **Vector Database**: Stored embeddings in Pinecone vector database accessed via Pinecone API
* **Content Metadata**: Created detailed JSON descriptions for the columns title, rating, vote_count, popularity, release_date, original_language, overview, genres, production_companies, cast, keywords, budget, collection, directors, imdb_id, country_of_origin, seasons_count, episode_count, runtime, network, etc.

**Recommendation Engine:**
* Uses Pinecone's default similarity search functionality with popularity weighting
* Advanced prompt engineering for LLM-generated recommendations
* Text-based user input queries matched against embeddings
* Recommendations with corresponding JSON descriptions passed to LLM for natural language response generation

### Q&A System

The Q&A system combines two approaches into a unified solution:

#### Part 1: Embeddings-Based Q&A
* **Data Source**: Uses plots.csv containing imdb_id, title, plot, and wikipedia_introduction
* **Embeddings**: Created sentence transformers embeddings for plots.csv and stored in Pinecone
* **Query Processing**: Extracts title information from text-based queries
* **Response Generation**: Combines content descriptions from recommendation system with plot and introduction info for comprehensive LLM responses

#### Part 2: Multi-Agent Web-Enhanced System
Built using **LangGraph** and **LangChain** frameworks with multiple specialized LLM models:

**Agent Architecture:**

1. **Query Database Agent**: Uses embedding-based system to provide initial answers
2. **Analyze Query Agent**: Determines if database answers are sufficient or if web search is required
3. **Web Search Agent**: Performs web searches using DuckDuckGo and scrapes information from top results
4. **Content Extraction Agent**: Extracts relevant information from scraped web content based on query context
5. **Response Synthesis Agent**: Creates final tailored responses using information from database agent and/or web search results


### Additional Functionality:
Managed follow-up queries, rewritten prompts for web search, and validated system relevance

## Implementation Details

### System Integration
* **Unified Interface**: Both recommendation and Q&A systems are combined under a single file that intelligently routes queries
* **API Layer**: Implemented using **FastAPI** for robust API endpoints
* **User Interface**: **Gradio**-based chatbot interface that calls FastAPI endpoints
* **Deployment**: Currently runs locally (not hosted)
* **Async Support**: Full async/await functionality throughout both systems
* **Docker Support**: Pull the docker image from **"darthvader640/movie-companion-app:latest"**

### Technical Stack
* **Embedding Model**: all-MPNET-base-v2
* **Vector Database**: Pinecone
* **Agent Framework**: LangGraph and LangChain
* **Web Search**: DuckDuckGo integration
* **Data Sources**: IMDb, Wikipedia, Web scraping
* **API Framework**: FastAPI
* **UI Framework**: Gradio
* **LLM Models**: Multiple models from Grow and Gemini were used based on specific strengths

## Features

* **Intelligent Query Routing**: Automatically determines whether to use recommendation or Q&A system
* **Comprehensive Movie Database**: 10,000+ titles with detailed metadata
* **Multi-Modal Q&A**: Combines local knowledge base with real-time web search
* **Natural Language Interface**: Text-based conversational queries
* **Context-Aware Responses**: Maintains conversation flow and handles follow-ups
* **Popularity-Weighted Recommendations**: Balances similarity with content popularity
* **Robust Fallback System**: Web search ensures comprehensive coverage for unknown queries

## Getting Started

### Prerequisites
```bash
# Required dependencies
pip install -r requirements.txt
```

### Installation
1. Clone the repository
2. Install dependencies
3. Set up Pinecone API credentials
4. Configure LLM API keys
5. Run the FastAPI server
6. Launch Gradio interface

Or run it directly by pulling the docker image and creating a container

### Docker Deployment

The application is containerized and available on Docker Hub:

```bash
# Pull the Docker image
docker pull darthvader640/movie-companion-app:latest

# Run using Docker Compose (recommended)
docker-compose up

# Or run manually
docker run -p 8000:8000 -p 7860:7860 darthvader640/movie-companion-app:latest
```

**Docker Configuration:**
- **FastAPI**: Runs on localhost - port:8000
- **Gradio UI**: Runs on localhost - port:7860
- **Dockerfile**: Optimized build with .dockerignore
- **Docker Compose**: Configured for both services

**Files included:**
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-service orchestration
- `.dockerignore` - Build optimization

## API Endpoints

* **POST /chat** - Unified chatbot interface
* **GET /health** - Get server response
* **DELETE /sessions** - Delete current session

## Future Enhancements

* **Hosting & Deployment**: Move from local to cloud-based hosting
* **User Preference Learning**: Track and adapt to individual user preferences
* **Enhanced UI/UX**: Improve chatbot interface from gradio to more advance frameworks
* **Real-time Data Updates**: Automated IMDb data refresh pipeline
* **Voice Interface**: Add speech-to-text and text-to-speech capabilities
* **Performance Optimization**: Improving the response time of each query

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
