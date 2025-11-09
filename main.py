import os
from typing import Optional, Dict, List, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from datetime import timedelta, datetime as dt
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import warnings
from Movie_Companian import MovieChatbot
import langchain
from langchain.cache import InMemoryCache
from langchain.memory import ConversationBufferMemory
from utils import APIRateLimiter
from summary_generator import generate_ai_summary, get_summary_generator
from dotenv import load_dotenv
import os
import httpx
import hashlib
import time
# Load .env and also try 'env' file if present
load_dotenv()
load_dotenv('env')

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/query about movies or TV shows")
    session_id: Optional[str] = Field(None, description="Optional session ID to maintain conversation context")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID for conversation tracking")
    query_type: Optional[str] = Field(None, description="Type of query: 'qa' or 'recommend'")
    is_movie_related: bool = Field(..., description="Whether the query was movie-related")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None



class Chatbot(MovieChatbot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        self.chat_history_limit = kwargs.get('chat_history_limit', 15)

    def get_memory(self, session_id: str = None) -> tuple[str, ConversationBufferMemory]:
        """Get or create session memory - simplified approach"""
        if not session_id:
            session_id = "default"
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                k=self.chat_history_limit
            )
        
        return session_id, self.sessions[session_id]

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list:
        """List all active session IDs"""
        return list(self.sessions.keys())


# Global chatbot instance
chatbot = None

# TMDB response cache for better performance
TMDB_CACHE = {}
TMDB_CACHE_TTL = 300  # 5 minutes

# Rate limiter instance
rate_limiter = APIRateLimiter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    global chatbot
    try:
        langchain.llm_cache = InMemoryCache()
        logger.info("LangChain cache initialized")

        chatbot = Chatbot(
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key"),
            groq_api_key=os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
        )
        logger.info("MovieChatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MovieChatbot: {e}")
        chatbot = None  # Allow API to start; /chat will return 503 until configured
    
    yield
    langchain.llm_cache = None
    logger.info("Application shutting down")

# Initialize FastAPI app 
app = FastAPI(
    title="Movie Chatbot API",
    description="A conversational AI API for movie and TV show information and recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# External API configuration (server-side)
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_API_KEY = os.getenv("TMDB_API_KEY") or os.getenv("tmdb_api_key")
TMDB_BEARER = os.getenv("TMDB_BEARER_TOKEN") or os.getenv("tmdb_bearer_token")
OMDB_BASE_URL = os.getenv("OMDB_BASE_URL", "http://www.omdbapi.com/")
OMDB_API_KEY = os.getenv("OMDB_API_KEY") or os.getenv("omdb_api_key")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or os.getenv("deepgram_api_key")

# Simple auth + persistence (SQLite)
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "dev_secret_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    # Connection pooling for better performance
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Disable SQL logging in production
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    chats = relationship("ChatSession", back_populates="owner")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.utcnow)
    messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete")
    owner = relationship("User", back_populates="chats")

class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' | 'bot'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=dt.utcnow)
    session = relationship("ChatSession", back_populates="messages")

class Title(Base):
    __tablename__ = "titles"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)  # 'tmdb' | 'omdb'
    external_id = Column(String, index=True)  # tmdb id or OMDB imdbID
    title = Column(String, index=True)
    original_title = Column(String)
    media_type = Column(String)
    year = Column(Integer)
    language = Column(String)
    popularity = Column(Float)
    vote_average = Column(Float)
    raw_json = Column(Text)
    updated_at = Column(DateTime, default=dt.utcnow)

class ContentSummary(Base):
    __tablename__ = "content_summaries"
    id = Column(Integer, primary_key=True, index=True)
    title_id = Column(Integer, ForeignKey("titles.id"), nullable=True)
    title_source = Column(String, nullable=True)
    title_external_id = Column(String, nullable=True)
    kind = Column(String, nullable=False)  # 'llm' | 'card' | 'plot' | 'llm_chat'
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=dt.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = dt.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

def get_cached_summary(db: Session, external_id: str, source: str, kind: str = 'llm') -> Optional[str]:
    """
    Get cached summary from database
    """
    try:
        summary = db.query(ContentSummary).filter(
            ContentSummary.title_external_id == external_id,
            ContentSummary.title_source == source,
            ContentSummary.kind == kind
        ).first()
        return summary.text if summary else None
    except Exception as e:
        print(f"Error retrieving cached summary: {e}")
        return None

def save_cached_summary(db: Session, external_id: str, source: str, text: str, kind: str = 'llm'):
    """
    Save summary to database cache
    """
    try:
        # Check if summary already exists
        existing = db.query(ContentSummary).filter(
            ContentSummary.title_external_id == external_id,
            ContentSummary.title_source == source,
            ContentSummary.kind == kind
        ).first()
        
        if existing:
            # Update existing summary
            existing.text = text
            existing.created_at = dt.utcnow()
        else:
            # Create new summary
            summary = ContentSummary(
                title_external_id=external_id,
                title_source=source,
                kind=kind,
                text=text
            )
            db.add(summary)
        
        db.commit()
        return True
    except Exception as e:
        print(f"Error saving cached summary: {e}")
        db.rollback()
        return False

def save_chat_batch(db: Session, user_id: int, session_id: str, messages: List[Tuple[str, str]]):
    """
    Efficiently save chat messages in batch to reduce database operations
    """
    try:
        # Get or create chat session in single query
        cs = db.query(ChatSession).filter(
            ChatSession.session_id == session_id, 
            ChatSession.user_id == user_id
        ).first()
        
        if cs is None:
            cs = ChatSession(user_id=user_id, session_id=session_id, title=None)
            db.add(cs)
            db.flush()  # Get ID without committing
        
        # Batch insert all messages
        chat_messages = [
            ChatMessageModel(chat_session_id=cs.id, role=role, content=content)
            for role, content in messages
        ]
        db.add_all(chat_messages)
        
        # Single commit for all operations
        db.commit()
        return cs
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error in batch chat save: {e}")
        raise e

def create_cache_key(url: str, params: dict) -> str:
    """Create a consistent cache key for TMDB requests"""
    key_data = f"{url}:{sorted(params.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()

def get_cached_response(cache_key: str):
    """Get cached TMDB response if valid"""
    if cache_key in TMDB_CACHE:
        response_data, timestamp = TMDB_CACHE[cache_key]
        if time.time() - timestamp < TMDB_CACHE_TTL:
            return response_data
        else:
            # Remove expired entry
            del TMDB_CACHE[cache_key]
    return None

def cache_response(cache_key: str, data):
    """Cache TMDB response with timestamp"""
    TMDB_CACHE[cache_key] = (data, time.time())
    
    # Periodic cleanup - remove old entries every 100 requests
    if len(TMDB_CACHE) % 100 == 0:
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in TMDB_CACHE.items()
            if current_time - timestamp > TMDB_CACHE_TTL
        ]
        for key in expired_keys:
            del TMDB_CACHE[key]

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        message="Movie Chatbot API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/auth/signup", response_model=TokenResponse)
async def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, hashed_password=get_password_hash(payload.password), display_name=payload.display_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)

@app.post("/auth/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)

@app.get("/me")
async def me(current_user: User = Depends(get_current_user)):
    return {"email": current_user.email, "display_name": current_user.display_name}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Main chat endpoint for movie-related queries"""
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        # Apply rate limiting for LLM calls
        await rate_limiter.wait_if_needed("groq")
        
        result = await chatbot.chat(request.message, request.session_id)
        
        # Handle case where result is a string instead of dict
        if isinstance(result, str):
            session_id, _ = chatbot.get_memory(request.session_id)
            
            # Use batch save for better performance
            messages = [('user', request.message), ('bot', result)]
            save_chat_batch(db, current_user.id, session_id, messages)
            
            # Also save as an LLM summary snippet if it looks like a title summary
            try:
                if request.message.lower().startswith('tell me about'):
                    db.add(ContentSummary(title_id=None, title_source=None, title_external_id=None, kind='llm_chat', text=result))
                    db.commit()
            except Exception:
                pass
            return ChatResponse(
                response=result,
                session_id=session_id,
                query_type=None,
                is_movie_related=True
            )
        
        # Persist using batch operations
        session_id = result["session_id"]
        messages = [('user', request.message), ('bot', result["response"])]
        save_chat_batch(db, current_user.id, session_id, messages)
        
        # Save response text as LLM summary snippet
        try:
            db.add(ContentSummary(title_id=None, title_source=None, title_external_id=None, kind='llm_chat', text=result["response"]))
            db.commit()
        except Exception:
            pass

        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            query_type=result.get("query_type"),
            is_movie_related=result["is_movie_related"]
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat/demo", response_model=ChatResponse)
async def demo_chat_endpoint(request: ChatRequest):
    """Demo chat endpoint without authentication for testing"""
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        # Apply rate limiting for LLM calls
        await rate_limiter.wait_if_needed("groq")
        
        result = await chatbot.chat(request.message, request.session_id)
        
        # Handle case where result is a string instead of dict
        if isinstance(result, str):
            session_id, _ = chatbot.get_memory(request.session_id)
            return ChatResponse(
                response=result,
                session_id=session_id,
                query_type=None,
                is_movie_related=True
            )
        
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            query_type=result.get("query_type"),
            is_movie_related=result.get("is_movie_related", True)
        )
        
    except Exception as e:
        logger.error(f"Error in demo chat endpoint: {e}")
        return ChatResponse(
            response="Sorry, I'm having trouble processing your request right now. Please try again later.",
            session_id=request.session_id or "demo_session",
            query_type="error",
            is_movie_related=False
        )

class SummaryRequest(BaseModel):
    external_id: str = Field(..., description="TMDB/OMDB external ID")
    title: str = Field(..., description="Movie/TV show title")
    year: Optional[str] = Field(None, description="Release year")
    media_type: str = Field(..., description="movie or tv")
    source: str = Field("tmdb", description="tmdb or omdb")

@app.post("/summary/generate")
async def generate_summary(
    request: SummaryRequest,
    db: Session = Depends(get_db)
):
    """Generate and cache movie/TV show summary using dedicated LLM with TMDB plot data"""
    try:
        # Check for cached summary first
        cached = get_cached_summary(db, request.external_id, request.source, 'llm')
        if cached:
            return {"summary": cached, "cached": True}

        # Generate new summary using dedicated LLM with TMDB data
        summary_gen = get_summary_generator()
        if summary_gen is None:
            raise HTTPException(status_code=503, detail="Summary generator not initialized. Check GROQ API key.")

        # Apply rate limiting
        await rate_limiter.wait_if_needed("groq")

        # Generate comprehensive summary using TMDB plot data
        summary = await generate_ai_summary(
            external_id=request.external_id,
            title=request.title,
            media_type=request.media_type,
            summary_type="detailed"  # Detailed summary for this endpoint
        )

        # Only cache and return valid summaries
        if summary:
            save_cached_summary(db, request.external_id, request.source, summary, 'llm')
            return {"summary": summary, "cached": False}
        else:
            raise HTTPException(status_code=503, detail="Unable to generate summary at this time")

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.get("/tmdb/{full_path:path}")
async def tmdb_proxy(full_path: str, request: Request):
    """Proxy selected TMDB GET endpoints through the backend to hide API keys with caching."""
    if not TMDB_API_KEY and not TMDB_BEARER:
        raise HTTPException(status_code=500, detail="TMDB credentials not configured on server")

    # Build target URL
    target_url = f"{TMDB_BASE_URL}/{full_path.lstrip('/') }"

    # Forward query params and include api_key if no bearer
    params = dict(request.query_params)
    if TMDB_API_KEY and 'api_key' not in params and not TMDB_BEARER:
        params['api_key'] = TMDB_API_KEY

    # Check cache first
    cache_key = create_cache_key(target_url, params)
    cached_data = get_cached_response(cache_key)
    if cached_data is not None:
        return cached_data

    # Apply rate limiting
    await rate_limiter.wait_if_needed("tmdb")

    headers = {"accept": "application/json"}
    if TMDB_BEARER:
        headers["Authorization"] = f"Bearer {TMDB_BEARER}"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(target_url, params=params, headers=headers)
            data = resp.json()
            
            # Cache the response
            cache_response(cache_key, data)
            # Persist common title structures for future embedding ingestion
            try:
                from sqlalchemy.orm import Session as _S
                db: _S = SessionLocal()
                def upsert(item):
                    ext_id = str(item.get('id')) if item.get('id') is not None else None
                    if not ext_id:
                        return
                    rec = db.query(Title).filter(Title.source == 'tmdb', Title.external_id == ext_id).first()
                    raw = item.copy()
                    payload = {
                        'source': 'tmdb',
                        'external_id': ext_id,
                        'title': item.get('title') or item.get('name') or '',
                        'original_title': item.get('original_title') or item.get('original_name') or '',
                        'media_type': item.get('media_type') or ('tv' if item.get('name') and not item.get('title') else 'movie'),
                        'year': int(str((item.get('release_date') or item.get('first_air_date') or '')).split('-')[0]) if (item.get('release_date') or item.get('first_air_date')) else None,
                        'language': item.get('original_language'),
                        'popularity': item.get('popularity') or 0.0,
                        'vote_average': item.get('vote_average') or 0.0,
                        'raw_json': json.dumps(raw),
                        'updated_at': dt.utcnow()
                    }
                    if rec:
                        for k,v in payload.items(): setattr(rec, k, v)
                    else:
                        rec = Title(**payload)
                        db.add(rec)
                # Results array
                import json
                if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
                    for it in data['results']:
                        upsert(it)
                # Single item detail
                elif isinstance(data, dict) and data.get('id') is not None:
                    upsert(data)
                db.commit()
                db.close()
            except Exception as _:
                pass
            return data
    except httpx.HTTPError as e:
        logger.error(f"TMDB proxy error: {e}")
        raise HTTPException(status_code=502, detail="Upstream TMDB error")

@app.get("/omdb")
async def omdb_proxy(request: Request):
    """Proxy OMDB queries (e.g., t=<title>) via server to hide API key with caching."""
    if not OMDB_API_KEY:
        raise HTTPException(status_code=500, detail="OMDB API key not configured on server")

    params = dict(request.query_params)
    params['apikey'] = OMDB_API_KEY
    
    # Check cache first
    cache_key = create_cache_key(OMDB_BASE_URL, params)
    cached_data = get_cached_response(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Apply rate limiting
    await rate_limiter.wait_if_needed("omdb")
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(OMDB_BASE_URL, params=params)
            data = resp.json()
            
            # Cache the response
            cache_response(cache_key, data)
            # Persist OMDB title for future embedding ingestion
            try:
                from sqlalchemy.orm import Session as _S
                db: _S = SessionLocal()
                if isinstance(data, dict) and data.get('Response') == 'True':
                    import json
                    imdb_id = data.get('imdbID')
                    if imdb_id:
                        rec = db.query(Title).filter(Title.source == 'omdb', Title.external_id == imdb_id).first()
                        payload = {
                            'source': 'omdb',
                            'external_id': imdb_id,
                            'title': data.get('Title') or '',
                            'original_title': data.get('Title') or '',
                            'media_type': 'movie',
                            'year': int(data.get('Year')) if (data.get('Year') and str(data.get('Year')).isdigit()) else None,
                            'language': data.get('Language'),
                            'popularity': 0.0,
                            'vote_average': float(data.get('imdbRating')) if data.get('imdbRating') not in (None, 'N/A') else 0.0,
                            'raw_json': json.dumps(data),
                            'updated_at': dt.utcnow()
                        }
                        if rec:
                            for k,v in payload.items(): setattr(rec, k, v)
                        else:
                            rec = Title(**payload)
                            db.add(rec)
                        db.commit()
                db.close()
            except Exception as _:
                pass
            return data
    except httpx.HTTPError as e:
        logger.error(f"OMDB proxy error: {e}")
        raise HTTPException(status_code=502, detail="Upstream OMDB error")

@app.get("/voice/health")
async def voice_health():
    """Report whether Deepgram-based STT/TTS is available."""
    has_deepgram = bool(DEEPGRAM_API_KEY)
    return {
        "deepgram": has_deepgram,
        "stt": has_deepgram,
        "tts": has_deepgram
    }

@app.post("/voice/tts")
async def voice_tts(payload: Dict[str, str]):
    """Text-to-speech via Deepgram Speak API. Returns audio/wav.
    Body: {"text": str, "model": str?, "encoding": str?, "container": str?}
    """
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=503, detail="Deepgram not configured")
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    model = payload.get("model", "aura-2-juno-en")
    encoding = payload.get("encoding", "linear16")
    container = payload.get("container", "wav")
    url = f"https://api.deepgram.com/v1/speak?model={model}&encoding={encoding}&container={container}"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "audio/wav"
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json={"text": text})
            if resp.status_code >= 400:
                msg = resp.text
                logger.error(f"Deepgram TTS error {resp.status_code}: {msg}")
                raise HTTPException(status_code=502, detail="Deepgram TTS failed")
            return Response(content=resp.content, media_type="audio/wav")
    except httpx.HTTPError as e:
        logger.error(f"Deepgram TTS HTTP error: {e}")
        raise HTTPException(status_code=502, detail="Deepgram TTS error")

from fastapi import UploadFile, File, Form
from fastapi.responses import Response

@app.post("/voice/stt")
async def voice_stt(
    audio: UploadFile | None = File(default=None),
    audio_base64: str | None = Form(default=None),
    mimetype: str | None = Form(default=None)
):
    """Speech-to-text via Deepgram Prerecorded Listen API.
    Accepts either multipart file "audio" or Form fields audio_base64 + mimetype.
    Returns: {"transcript": "..."}
    """
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=503, detail="Deepgram not configured")

    audio_bytes = None
    content_type = None
    if audio is not None:
        audio_bytes = await audio.read()
        content_type = audio.content_type or "audio/wav"
    elif audio_base64:
        import base64
        audio_bytes = base64.b64decode(audio_base64)
        content_type = mimetype or "audio/wav"
    else:
        raise HTTPException(status_code=400, detail="No audio provided")

    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type,
        "Accept": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, content=audio_bytes)
            if resp.status_code >= 400:
                logger.error(f"Deepgram STT error {resp.status_code}: {resp.text}")
                raise HTTPException(status_code=502, detail="Deepgram STT failed")
            data = resp.json()
            # Extract main transcript
            transcript = (
                data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
            )
            return {"transcript": transcript}
    except httpx.HTTPError as e:
        logger.error(f"Deepgram STT HTTP error: {e}")
        raise HTTPException(status_code=502, detail="Deepgram STT error")

@app.post("/summaries")
async def save_content_summary(payload: Dict[str, str], current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Optional endpoint to save arbitrary summaries (card descriptions, plots) for future embedding.
    Body: { text: str, title_source?: str, title_external_id?: str, kind?: 'llm'|'card'|'plot' }
    """
    text = (payload.get('text') or '').strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    kind = payload.get('kind') or 'card'
    ts = payload.get('title_source')
    te = payload.get('title_external_id')
    title_id = None
    if ts and te:
        rec = db.query(Title).filter(Title.source == ts, Title.external_id == te).first()
        if rec:
            title_id = rec.id
    db.add(ContentSummary(title_id=title_id, title_source=ts, title_external_id=te, kind=kind, text=text))
    db.commit()
    return {"status": "ok"}
    
@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Clear a specific chat session"""
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        success = chatbot.clear_session(session_id)
        # Also delete persisted session if exists
        cs = db.query(ChatSession).filter(ChatSession.session_id == session_id, ChatSession.user_id == current_user.id).first()
        if cs:
            db.delete(cs)
            db.commit()
        if success:
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions")
async def list_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all active sessions"""
    try:
        db_sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()
        output = []
        for s in db_sessions:
            output.append({
                "id": s.id,
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.isoformat()
            })
        return {"total_sessions": len(output), "sessions": output}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        cs = db.query(ChatSession).filter(ChatSession.session_id == session_id, ChatSession.user_id == current_user.id).first()
        if not cs:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = (
            db.query(ChatMessageModel)
            .filter(ChatMessageModel.chat_session_id == cs.id)
            .order_by(ChatMessageModel.created_at.asc())
            .all()
        )
        return {
            "session_id": cs.session_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in msgs
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session messages: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/trending/summaries/generate")
async def generate_trending_summaries(db: Session = Depends(get_db)):
    """Generate AI summaries for trending titles using dedicated LLM with TMDB plot data"""
    try:
        summary_gen = get_summary_generator()
        if summary_gen is None:
            raise HTTPException(status_code=503, detail="Summary generator not initialized. Check GROQ API key.")

        # Get trending endpoints to fetch from
        trending_endpoints = [
            '/trending/movie/day', '/trending/movie/week', '/movie/popular', '/movie/top_rated',
            '/trending/tv/day', '/trending/tv/week', '/tv/popular', '/tv/top_rated'
        ]

        generated_count = 0
        processed_ids = set()

        # Apply rate limiting
        await rate_limiter.wait_if_needed("tmdb")

        for endpoint in trending_endpoints:
            try:
                # Fetch trending data from TMDB
                async with httpx.AsyncClient(timeout=20) as client:
                    headers = {"accept": "application/json"}
                    params = {}
                    if TMDB_API_KEY:
                        params['api_key'] = TMDB_API_KEY
                    if TMDB_BEARER:
                        headers["Authorization"] = f"Bearer {TMDB_BEARER}"

                    target_url = f"{TMDB_BASE_URL}{endpoint}"
                    resp = await client.get(target_url, params=params, headers=headers)

                    if resp.status_code != 200:
                        continue

                    data = resp.json()

                    # Process each item
                    for item in (data.get('results', [])[:10]):  # Limit to top 10 for each category
                        external_id = str(item.get('id', ''))
                        if not external_id or external_id in processed_ids:
                            continue

                        processed_ids.add(external_id)

                        # Check if summary already exists
                        existing = get_cached_summary(db, external_id, 'tmdb', 'card')
                        if existing:
                            continue

                        # Generate summary
                        title = item.get('title') or item.get('name', '')
                        if not title:
                            continue

                        # Determine media type
                        media_type = 'tv' if item.get('name') else 'movie'

                        try:
                            # Apply rate limiting for LLM
                            await rate_limiter.wait_if_needed("groq")

                            # Generate summary using dedicated LLM with TMDB plot data
                            summary = await generate_ai_summary(
                                external_id=external_id,
                                title=title,
                                media_type=media_type,
                                summary_type="card"  # Card summary for trending display
                            )

                            # Only save valid summaries
                            if summary:
                                save_cached_summary(db, external_id, 'tmdb', summary, 'card')
                                generated_count += 1
                                logger.info(f"Generated summary for {title} (ID: {external_id})")
                            else:
                                logger.warning(f"Failed to generate valid summary for {title} (ID: {external_id})")

                            # Small delay to prevent overwhelming the LLM
                            if generated_count % 3 == 0:
                                import asyncio
                                await asyncio.sleep(0.5)

                        except Exception as e:
                            logger.warning(f"Failed to generate summary for {title}: {e}")
                            continue

            except Exception as e:
                logger.warning(f"Failed to process endpoint {endpoint}: {e}")
                continue

        return {"generated": generated_count, "message": f"Generated {generated_count} new trending summaries using TMDB plot data"}

    except Exception as e:
        logger.error(f"Error generating trending summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating summaries: {str(e)}")

@app.get("/trending/summaries/check")
async def check_trending_summaries(db: Session = Depends(get_db)):
    """Check which trending titles need AI summaries and generate them"""
    try:
        # This endpoint will be called when the app loads to ensure trending titles have summaries
        missing_summaries = []

        # Get recent titles from the database that don't have card summaries
        recent_titles = (
            db.query(Title)
            .filter(Title.source == 'tmdb')
            .filter(Title.popularity > 10)  # Only check popular titles
            .order_by(Title.updated_at.desc())
            .limit(50)
            .all()
        )

        for title in recent_titles:
            existing = get_cached_summary(db, title.external_id, 'tmdb', 'card')
            if not existing:
                missing_summaries.append({
                    'external_id': title.external_id,
                    'title': title.title,
                    'media_type': title.media_type,
                    'year': title.year
                })

        return {
            "missing_count": len(missing_summaries),
            "missing_summaries": missing_summaries[:10],  # Return first 10
            "needs_generation": len(missing_summaries) > 0
        }

    except Exception as e:
        logger.error(f"Error checking trending summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking summaries: {str(e)}")

@app.get("/trending/summaries/{external_id}")
async def get_trending_summary(external_id: str, db: Session = Depends(get_db)):
    """Get AI summary for a specific trending title"""
    try:
        # Check for card summary first
        summary = get_cached_summary(db, external_id, 'tmdb', 'card')
        if summary:
            return {"summary": summary, "cached": True}

        # If no card summary, check for general LLM summary
        summary = get_cached_summary(db, external_id, 'tmdb', 'llm')
        if summary:
            return {"summary": summary, "cached": True}

        return {"summary": None, "cached": False}

    except Exception as e:
        logger.error(f"Error getting trending summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info"
    )