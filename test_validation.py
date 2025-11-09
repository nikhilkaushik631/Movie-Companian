#!/usr/bin/env python3
"""
Test script for the new LLM-based query validation with conversation history
"""
import asyncio
import os
from dotenv import load_dotenv
from Movie_Companian import MovieChatbot

load_dotenv()

async def test_validation():
    """Test the enhanced validation logic"""

    # Initialize chatbot
    print("Initializing MovieChatbot...")
    chatbot = MovieChatbot()

    print("\n" + "="*80)
    print("Testing Enhanced Query Validation with Conversation History")
    print("="*80 + "\n")

    # Test 1: Direct movie query (should be valid)
    print("Test 1: Direct movie query")
    print("-" * 40)
    query1 = "Tell me about Inception"
    is_valid1 = await chatbot.validate_movie_query(query1)
    print(f"Query: '{query1}'")
    print(f"Valid: {is_valid1} ✓" if is_valid1 else f"Valid: {is_valid1} ✗")
    print(f"Expected: True\n")

    # Simulate conversation by adding to memory
    chatbot.memory.save_context(
        {"input": query1},
        {"output": "Inception is a 2010 sci-fi film directed by Christopher Nolan about dream invasion."}
    )

    # Test 2: Follow-up query with pronoun (should be valid due to history)
    print("Test 2: Follow-up query with pronoun")
    print("-" * 40)
    query2 = "Who directed it?"
    is_valid2 = await chatbot.validate_movie_query(query2)
    print(f"Query: '{query2}'")
    print(f"Valid: {is_valid2} ✓" if is_valid2 else f"Valid: {is_valid2} ✗")
    print(f"Expected: True (follow-up to previous movie discussion)")
    print(f"Conversation history: {len(chatbot.memory.load_memory_variables({}).get('chat_history', []))} messages\n")

    # Test 3: Another follow-up
    chatbot.memory.save_context(
        {"input": query2},
        {"output": "Inception was directed by Christopher Nolan."}
    )

    print("Test 3: Follow-up asking for more")
    print("-" * 40)
    query3 = "What about his other movies?"
    is_valid3 = await chatbot.validate_movie_query(query3)
    print(f"Query: '{query3}'")
    print(f"Valid: {is_valid3} ✓" if is_valid3 else f"Valid: {is_valid3} ✗")
    print(f"Expected: True (follow-up asking about director's other movies)")
    print(f"Conversation history: {len(chatbot.memory.load_memory_variables({}).get('chat_history', []))} messages\n")

    # Test 4: Completely unrelated query (should be invalid)
    print("Test 4: Unrelated query")
    print("-" * 40)
    query4 = "What's the weather forecast for tomorrow?"
    is_valid4 = await chatbot.validate_movie_query(query4)
    print(f"Query: '{query4}'")
    print(f"Valid: {is_valid4} ✗" if not is_valid4 else f"Valid: {is_valid4} ✓")
    print(f"Expected: False (unrelated to movies)\n")

    # Test 5: Generic follow-up without movie context (edge case)
    # Clear memory first
    chatbot.memory.clear()
    print("Test 5: Generic query without context")
    print("-" * 40)
    query5 = "Tell me more about that"
    is_valid5 = await chatbot.validate_movie_query(query5)
    print(f"Query: '{query5}'")
    print(f"Conversation history: {len(chatbot.memory.load_memory_variables({}).get('chat_history', []))} messages")
    print(f"Valid: {is_valid5}")
    print(f"Expected: Could be True or False (ambiguous without context)\n")

    # Test 6: Direct recommendation query
    print("Test 6: Recommendation query")
    print("-" * 40)
    query6 = "Recommend me some action movies"
    is_valid6 = await chatbot.validate_movie_query(query6)
    print(f"Query: '{query6}'")
    print(f"Valid: {is_valid6} ✓" if is_valid6 else f"Valid: {is_valid6} ✗")
    print(f"Expected: True (direct movie recommendation request)\n")

    print("="*80)
    print("Validation tests completed!")
    print("="*80)

    # Print stats
    stats = chatbot.load_balancer.get_stats()
    print(f"\nAPI Usage Statistics:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"API distribution: {stats['api_distribution']}")

if __name__ == "__main__":
    asyncio.run(test_validation())
