# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cinemizer is an entertainment companion chatbot that combines a **FastAPI backend** (Python) with a **Next.js 14 frontend** (TypeScript/React). The system uses RAG (Retrieval Augmented Generation) with Pinecone vector database and LangGraph multi-agent architecture to provide movie/TV recommendations and answer entertainment questions.

## Development Commands

### Backend (Python/FastAPI)

**Start backend server:**
```bash
# Development
python main.py

# With uvicorn directly
uvicorn main:app --reload --port 8000
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Environment setup:**
- Copy `.env.template` to `.env` and configure API keys (Google/Gemini, Groq, TMDB, OMDB, Pinecone, Deepgram)
- Backend loads from both `.env` and `env` files

### Frontend (Next.js)

**Development server:**
```bash
# From root directory
npm run dev              # Runs on port 3000
npm run dev:demo         # Demo mode on port 3000
npm run dev:demo:3001    # Demo mode on port 3001
npm run dev:demo:3002    # Demo mode on port 3002

# From frontend-next directory
cd frontend-next
npm run dev
```

**Build and production:**
```bash
npm run build    # Next.js production build
npm run start    # Start production server
npm run lint     # Run ESLint
```

### Docker

**Local deployment:**
```bash
# Quick deploy script
./deploy.sh

# Or manually
docker-compose up --build

# Production stack
docker-compose -f docker-compose.production.yml up --build
```

**Access points:**
- Frontend: `http://localhost:3000` (dev) or `http://localhost:8080` (docker with nginx)
- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Architecture

### Backend Structure

**Core System Files:**
- `main.py` - FastAPI application, authentication (JWT), database (SQLAlchemy), session management, API endpoints
- `Movie_Companian.py` - Main chatbot orchestrator with `APILoadBalancer` for multi-LLM routing
- `Multi_Agent.py` - LangGraph-based multi-agent Q&A system with 5 specialized agents
- `Content_Recommend.py` - RAG-based recommendation engine using Pinecone embeddings
- `QA.py` - Embeddings-based Q&A using plots and Wikipedia data

**Supporting Modules:**
- `utils.py` - `LRUCache`, `APIRateLimiter` utilities
- `summary_generator.py` - AI-powered summaries for trending content
- `trending_summaries.py` - Trending content management
- `tts.py` - Text-to-speech functionality

**Legacy Files:**
- `app.html` - Original HTML/JS UI (replaced by Next.js frontend)
- `Gradio_App.py`, `gradio.py` - Gradio interface (deprecated)

### Multi-Agent System (Multi_Agent.py)

LangGraph workflow with 5 specialized agents:
1. **Query Database Agent** - Searches Pinecone embeddings
2. **Analyze Query Agent** - Determines if web search needed
3. **Web Search Agent** - DuckDuckGo search + scraping
4. **Content Extraction Agent** - Extracts relevant info from web results
5. **Response Synthesis Agent** - Generates final response

### Frontend Structure (Next.js 14)

**App Router:**
- `app/page.tsx` - Landing/home page
- `app/chat/page.tsx` - Chat interface
- `app/login/page.tsx` - Authentication page
- `app/layout.tsx` - Root layout
- `app/globals.css` - Global styles

**Environment variable:**
- `NEXT_PUBLIC_DEMO=1` enables demo mode

### API Integration

**Main Endpoints:**
- `POST /chat` - Chat with AI (requires authentication)
- `POST /signup` - User registration
- `POST /token` - Login (returns JWT)
- `GET /health` - Health check
- `DELETE /sessions/{session_id}` - Clear chat session
- `GET /trending` - Get trending content with AI summaries
- `GET /search` - Search movies/shows

### LLM Load Balancing

The `APILoadBalancer` in `Movie_Companian.py` manages multiple LLM APIs:
- **Gemini**: 2 API keys, uses `gemini-2.0-flash` for fast tasks, `gemini-1.5-pro` for complex
- **Groq**: `llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `openai/gpt-oss-20b`
- **Hugging Face**: Optional `meta-llama/Llama-3.2-3B-Instruct` (if token available)

Three complexity tiers: `fast_models` (validation/routing), `standard_models` (recommendations), `advanced_models` (complex Q&A)

### Data Sources

- **Vector Database**: Pinecone (movie/TV embeddings)
- **Embeddings**: sentence-transformers (`all-MPNET-base-v2`, `all-MiniLM-L6-V2`)
- **Dataset**: 10,000+ titles scraped from IMDb (in `dataset/` directory)
- **Database**: SQLite (`app.db`) for user auth and chat history

### Authentication

JWT-based authentication with:
- `SECRET_KEY` and `ALGORITHM` environment variables (see `main.py:147-148`)
- OAuth2 password bearer tokens
- Password hashing with passlib/bcrypt

## Key Configuration

**Required Environment Variables:**
```
GOOGLE_API_KEY
GEMINI_API_KEY
GROQ_API_KEY
TMDB_API_KEY
OMDB_API_KEY
PINECONE_API_KEY
DEEPGRAM_API_KEY (optional)
HUGGINGFACE_API_TOKEN (optional)
SECRET_KEY (for JWT)
```

**Database:**
- Default: SQLite (`app.db`)
- Configured in `main.py:154` via `DATABASE_URL` env var

## Performance Notes

- LangChain uses `InMemoryCache` for LLM caching (see `Movie_Companian.py:18`)
- `APILoadBalancer` includes aggressive response caching (10-30 min TTL)
- Database connection pooling configured in `main.py` (pool_size=20, max_overflow=30)
- Session management uses `ConversationBufferMemory` with 15-message limit

## Deployment

See `DEPLOYMENT_GUIDE.md` for full deployment instructions. Quick options:
- **Local**: `./deploy.sh` (Docker Compose)
- **Production**: `docker-compose -f docker-compose.production.yml up`
- **Railway/Render**: Configuration files included (`railway.json`, `render.yaml`)
- **Vercel**: Frontend can deploy via `vercel.json`

## Git Workflow

Current development branch: `claude/update-cla-011CUxBN59TM7DhbcXCyuG9J`

Always develop on the designated `claude/*` branch and push with:
```bash
git push -u origin <branch-name>
```
