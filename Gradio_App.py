import gradio as gr
import requests
from typing import List, Dict
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update this to match your FastAPI server
SESSION_ID = str(uuid.uuid4())  # Generate a unique session ID

def chat_with_bot(message: str, history: List[dict]) -> tuple[str, List[dict]]:
    """
    Send message to the chatbot API and return response with updated history
    """
    try:
        # Prepare the request payload
        payload = {
            "message": message,
            "session_id": SESSION_ID
        }
        
        # Make API call
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            bot_response = result["response"]
            
            # Update history with OpenAI-style messages
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            
            return "", history
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
            
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to the chatbot server. Make sure your FastAPI server is running on localhost:8000"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. The server might be busy."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history

def clear_chat():
    """Clear the chat history"""
    try:
        # Clear session on server
        requests.delete(f"{API_BASE_URL}/sessions/{SESSION_ID}")
    except:
        pass  # Ignore errors when clearing
    
    return []

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return "Server is running"
        else:
            return f"Server responded with status: {response.status_code}"
    except:
        return "Server is not accessible"

# Create the Gradio interface
with gr.Blocks(
    title="Movie Chatbot",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    """
) as demo:
    
    gr.Markdown("# Movie Companion")
    gr.Markdown("Ask me anything about movies, TV shows, or get personalized recommendations!")
    
    # Server status indicator
    status_display = gr.Textbox(
        label="Server Status",
        value=check_server_status(),
        interactive=False,
        max_lines=1,
        elem_classes="status-box"
    )
    
    # Chat interface
    chatbot = gr.Chatbot(
        height=400,
        show_label=False,
        container=True,
        type="messages",
        layout="bubble",
        elem_id="chatbot"
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about movies, tv shows or get recommendations...",
            show_label=False,
            scale=4,
            container=False
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Row():
        clear_btn = gr.Button("Clear Chat", scale=1)
        refresh_status_btn = gr.Button("Refresh Status", scale=1)
    
    # Event handlers
    msg.submit(
        chat_with_bot,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    send_btn.click(
        chat_with_bot,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot]
    )
    
    refresh_status_btn.click(
        check_server_status,
        outputs=[status_display]
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            "recommend some good sci-fi movies from the 2020s?",
            "Tell me about the movie Inception",
            "Recommend a comedy movie for tonight",
            "Who directed The Dark Knight?",
            "What are the best HBO series right now?"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    print(" Starting Gradio Movie Chatbot Interface...")
    print(" Make sure your FastAPI server is running on http://localhost:8000")
    print(" Gradio interface will be available at: http://localhost:7860")
    
    demo.launch(
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_api=False
    )