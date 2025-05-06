# Smart Entertainment Assistant using RAG and AI Agents

A powerful entertainment assistant that provides movie/TV recommendations and answers questions about entertainment content using Retrieval Augmented Generation (RAG) and AI agents.

## Project Overview

This project consists of two main components:

1. **Recommendation System** - Provides personalized movie and TV show recommendations based on user queries
2. **Q&A and Trivia System** - Answers questions about movies, TV shows, and entertainment topics

## Architecture

### Recommendation System

- **Data Collection**: Scraped IMDb for 10,000+ movies and TV shows (worldwide and Indian content)
- **Embedding Generation**: Used BERT (MiniLM-L6-v2) to create embeddings of content descriptions
- **Vector Database**: Stored embeddings in ChromaDB for similarity searching
- **Content Metadata**: Created detailed JSON descriptions including title, rating, popularity, cast, genres, etc.
- **Recommendation Engine**: Uses vector similarity search with popularity weighting
- **Response Generation**: LLM generates natural language recommendations based on retrieved content

### Q&A System

#### Embeddings-Based Q&A

- Uses the same content embeddings as the recommendation system
- Extracts title from user query and retrieves relevant information
- Combines plot data, Wikipedia introductions, and content metadata
- LLM generates comprehensive answers based on retrieved information

#### Multi-Agent Web-Enhanced Q&A

Built using LangGraph and LangChain frameworks with multiple specialized agents:

1. **Query Analysis Agent**: Determines if a query is within the system's scope
2. **Database Query Agent**: Uses embedding-based retrieval for answers
3. **Analysis Agent**: Determines if database answers are sufficient or web search is needed
4. **Web Search Agent**: Searches DuckDuckGo and retrieves top results
5. **Content Extraction Agent**: Extracts relevant information from web results
6. **Response Synthesis Agent**: Creates final, tailored responses

## Features

- Text-based natural language interface
- Personalized entertainment recommendations
- Detailed answers to movie/TV show questions
- Seamless web search integration for unknown information
- Context-aware conversation with history tracking
- Follow-up question handling

## Technical Implementation

- **Embedding Model**: BERT (MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **Agent Framework**: LangGraph and LangChain
- **Search Integration**: DuckDuckGo
- **Data Sources**: IMDb, Wikipedia, Web

## Future Work

- User preference tracking
- Voice interface integration
- Improved multi-language support
- Personalized recommendation fine-tuning

 
