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
- `Movie_Companian.py` - Main chatbot orchestrator with:
  - `APILoadBalancer` for multi-LLM routing
  - Greeting detection and friendly introductions
  - LLM-based query validation with conversation history support
  - Follow-up query detection using context
- `Multi_Agent.py` - LangGraph-based multi-agent Q&A system with temporal pattern detection
- `Content_Recommend.py` - RAG-based recommendation engine using Pinecone embeddings
- `QA.py` - Embeddings-based Q&A using plots and Wikipedia data

**Supporting Modules:**
- `utils.py` - `LRUCache`, `APIRateLimiter` utilities
- `summary_generator.py` - AI-powered summaries with:
  - Enhanced prompts to prevent LLM from asking questions
  - Response validation to filter out question patterns
  - Increased token limit (250) for better quality
  - Cache validation before saving
- `trending_summaries.py` - Trending content management
- `tts.py` - Text-to-speech functionality

**Legacy Files:**
- `app.html` - Original HTML/JS UI (replaced by Next.js frontend)
- `Gradio_App.py`, `gradio.py` - Gradio interface (deprecated)

### Multi-Agent System (Multi_Agent.py)

LangGraph workflow with 5 specialized agents:
1. **Query Database Agent** - Searches Pinecone embeddings
2. **Analyze Query Agent** - Determines if web search needed with temporal pattern detection
3. **Web Search Agent** - DuckDuckGo search + scraping
4. **Content Extraction Agent** - Extracts relevant info from web results
5. **Response Synthesis Agent** - Generates final response

**Temporal Pattern Detection:**
- Automatically routes queries about recent content to web search instead of database
- Detects year patterns (2020-2029) and temporal phrases (this month, last week, this year, etc.)
- Ensures up-to-date results for queries like "movies from this month" or "2024 releases"
- Patterns detected: year ranges, month names, temporal references (see `Multi_Agent.py:158-171`)

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
  - Handles greetings with friendly introductions
  - Validates queries using LLM with conversation history
  - Detects follow-up questions based on context
- `POST /signup` - User registration
- `POST /token` - Login (returns JWT)
- `GET /health` - Health check
- `DELETE /sessions/{session_id}` - Clear chat session
- `GET /trending` - Get trending content with AI summaries
- `GET /search` - Search movies/shows
- `DELETE /summary/cache/clear` - Clear all cached summaries from database
- `GET /tmdb/{full_path}` - Proxy TMDB API endpoints with caching

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

## Recent Improvements

### Query Routing & Temporal Detection
- **Temporal pattern detection** for recent content queries (see `Multi_Agent.py:158-171`)
  - Detects year patterns (2020-2029) and temporal phrases ("this month", "last week", etc.)
  - Routes queries about recent timelines to web search instead of database
  - Fixes issue where "movies from this month" returned outdated results

### Greeting & Introduction Handling
- **Smart greeting detection** (see `Movie_Companian.py:621-643`)
  - Recognizes common greetings: hi, hello, hey, greetings, etc.
  - Responds with friendly introduction and capabilities overview
  - Properly saves greetings to conversation history for context
  - Returns query_type: "greeting" for tracking

### Enhanced AI Summary Generation
- **Improved quality control** (see `summary_generator.py:61-82`)
  - Enhanced prompts to prevent LLM from asking questions back
  - Response validation filters out question patterns and invalid responses
  - Increased token limit from 200 to 250 for more comprehensive summaries
  - Cache validation before saving to prevent bad summaries
  - Frontend cache versioning system for easy invalidation

### Cache Management Features
- **Summary cache clearing endpoint** (see `main.py:582-598`)
  - `DELETE /summary/cache/clear` clears all cached summaries
  - Useful for invalidating old/bad summaries after quality improvements
  - Returns count of deleted summaries for verification

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

### Free Tier Deployment (Recommended)

**Quick Start** (5 minutes):
- See [QUICK_DEPLOY.md](QUICK_DEPLOY.md) for fastest deployment
- Frontend: Vercel (free, unlimited bandwidth)
- Backend: Railway.app (free $5/month credit)

**Detailed Guides:**
- [FREE_DEPLOYMENT_GUIDE.md](FREE_DEPLOYMENT_GUIDE.md) - Complete free tier deployment guide
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Step-by-step checklist
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Full deployment options

**Configuration Files:**
- `vercel.json` - Vercel deployment configuration
- `frontend-next/vercel.json` - Frontend-specific Vercel config
- `railway.json` - Railway.app configuration
- `render.yaml` - Render.com configuration
- `Dockerfile.backend` - Optimized backend container
- `start.sh` - Production startup script

**Local Development:**
- `./deploy.sh` - Quick local Docker Compose deployment
- `docker-compose.yml` - Local development stack
- `docker-compose.production.yml` - Production-ready stack

## Git Workflow

Current development branch: `claude/update-cla-011CUxBN59TM7DhbcXCyuG9J`

Always develop on the designated `claude/*` branch and push with:
```bash
git push -u origin <branch-name>
```
