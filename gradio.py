import gradio as gr
from Movie_Companian import MovieChatbot 


# For simplicity in Gradio, we'll initialize it once.
try:
    chatbot_instance = MovieChatbot()
    print("MovieChatbot initialized successfully for Gradio.")
except Exception as e:
    print(f"Error initializing MovieChatbot: {e}")
    print("Please ensure your API keys are set as environment variables or passed correctly.")
    chatbot_instance = None

# --- Gradio Interface Logic ---
async def movie_chat_interface(user_input: str, history: list):
    """
    This function will be called by Gradio for each user input.
    It interacts with your MovieChatbot instance and streams updates.
    """
    if not user_input.strip():
        yield history, "" 
        return

    if chatbot_instance is None:
        history.append([user_input, "Error: Chatbot not initialized. Check API keys."])
        yield history, ""
        return

    if user_input.lower() == 'clear':
        if chatbot_instance and hasattr(chatbot_instance, 'memory') and hasattr(chatbot_instance.memory, 'clear'):
            chatbot_instance.memory.clear()
            print("Chat history cleared.")
        yield [], ""
        return


    try:
        history.append([user_input, "Thinking...."])  
        yield history, ""

        response_text = await chatbot_instance.chat(user_input)
        
        history[-1][1] = response_text
        yield history, ""
        
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        print(f"Error during chatbot response generation: {e}")
        history[-1][1] = error_msg
        yield history, ""

# # --- Create the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Entertainment Companion") as demo:
    gr.Markdown(
        """
        # 🎬 Entertainment Companion Chatbot 🍿
        Ask me anything about **Movies** and **TV Shows**, or ask for **Recommendations**!
        """
    )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=360,
        avatar_images=("https://img.icons8.com/color/48/user.png", "https://img.icons8.com/fluency/48/chatbot.png"),
        show_copy_button=True,  
        editable="user"
    )
    

    # Textbox for user input and button for submission

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter your movie or TV show query...",
            container=False,
        )
        submit_btn = gr.Button("SUBMIT", variant="primary", scale=0, icon="https://img.icons8.com/fluency/24/filled-sent.png")
        clear_btn = gr.Button("CLEAR", variant="secondary", scale=0, icon="https://img.icons8.com/fluency/24/delete-forever.png")


    with gr.Column():
        gr.Examples(
            examples=[
                "Recommend me a sci-fi movie from the 2020s",
                "What's the plot of Inception?",
                "Best horror movies of all time",
                "Shows like prison break"
            ],
            inputs=txt,
            label=""
        )


    # Custom clear function that also clears the chatbot's memory
    def clear_chat():
        if chatbot_instance:
            chatbot_instance.memory.clear()
        return [], ""
    

    # Connect the components: when user types and clicks send or presses Enter
    txt.submit(movie_chat_interface, [txt, chatbot], [chatbot, txt], queue=True)
    submit_btn.click(movie_chat_interface, [txt, chatbot], [chatbot, txt], queue=True)
    clear_btn.click(clear_chat, outputs=[chatbot, txt])



# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(debug=True) 