import os
from typing import Optional, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import warnings
from Movie_Companian import MovieChatbot
import langchain
from langchain.cache import InMemoryCache
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    global chatbot
    try:
        langchain.llm_cache = InMemoryCache()
        logger.info("LangChain cache initialized")

        chatbot = Chatbot(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        logger.info("MovieChatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MovieChatbot: {e}")
        raise
    
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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for movie-related queries"""
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
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
            is_movie_related=result["is_movie_related"]
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific chat session"""
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        success = chatbot.clear_session(session_id)
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
async def list_sessions():
    """List all active sessions"""
    try:
        if chatbot is None:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        sessions = chatbot.list_sessions()
        return {
            "total_sessions": len(sessions),
            "active_sessions": sessions
        }
    
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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