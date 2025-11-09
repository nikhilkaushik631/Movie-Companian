# cinemizer

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
* **User Interface**: Custom HTML/JS UI (`app.html`) that calls FastAPI endpoints, with voice (STT/TTS)
* **Deployment**: Currently runs locally (not hosted)
* **Async Support**: Full async/await functionality throughout both systems
* **Docker Support**: Build and run locally via Docker

### Technical Stack
* **Embedding Model**: all-MPNET-base-v2 and all-MiniLM-L6-V2
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
3. Set up Pinecone (optional) and other API credentials
4. Configure LLM API keys via environment variables:

```bash
export GOOGLE_API_KEY=your_key
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
```

Or create a `.env` file with the same variables.

5. Run the FastAPI server
6. Open `app.html` in your browser

Or run it directly by pulling the docker image and creating a container

### Docker Deployment

Build and run locally (with reverse proxy and static hosting):

```bash
# Build
docker build -t movie-companion:local .

# Start stack (API + Nginx)
docker compose up --build
```

Open the app at `http://localhost:8080` (Nginx serves `app.html` and proxies `/api` to FastAPI).

**Docker Configuration:**
- **FastAPI**: Internal container port 8000 (service `api`)
- **Nginx**: Exposed on `http://localhost:8080` serving UI and proxying `/api`
- **Dockerfile**: Starts `uvicorn main:app`

**Files included:**
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-service orchestration
- `.dockerignore` - Build optimization

## API Endpoints

* **POST /chat** - Unified chatbot interface
* **GET /health** - Get server response
* **DELETE /sessions** - Delete current session

## Frontend Usage

- Open `app.html` in a modern browser.
- Chat section sends your messages to the FastAPI `/chat` endpoint.
- Clicking a trending card or search result now asks the AI about that title instead of showing a generic card.
- Toggle voice responses with the "🔊 Voice" button. The bot will speak its answers using the browser's speech synthesis.
- Use the mic button to dictate your query using the browser's speech recognition.

Permissions:
- On first use, the browser will ask for microphone permission for speech recognition.

Notes:
- Voice features use built-in browser APIs (Web Speech). They may be unavailable on some browsers/devices.
- Ensure the backend is running at `http://localhost:8000` (configurable in `app.html` as `apiBaseUrl`).

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
