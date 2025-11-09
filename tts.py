import gradio as gr
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions
from Movie_Companian import MovieChatbot
import tempfile
import soundfile as sf


# --- Load Environment Variables ---
# Make sure to have a .env file with your DEEPGRAM_API_KEY
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("deepgram_api_key")

# --- Initialize Deepgram and MovieChatbot ---
try:
    if DEEPGRAM_API_KEY is None:
        raise ValueError("DEEPGRAM_API_KEY not found in environment variables.")
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    chatbot_instance = MovieChatbot()
    print("MovieChatbot and DeepgramClient initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")
    print("Please ensure your API keys are set as environment variables.")
    chatbot_instance = None
    deepgram = None

# --- Core Chatbot & STT/TTS Logic ---

async def process_audio_and_chat(audio_input, history: list):
    """
    Handles STT transcription from audio input, then passes the text to the chat interface.
    """
    if audio_input is None:
        return history, "", None

    # Save audio to a temporary file to send to Deepgram
    samplerate, audio_data = audio_input
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_data, samplerate)
        tmp_path = tmpfile.name

    try:
        # STT: Transcribe audio using Deepgram
        with open(tmp_path, 'rb') as audio_file:
            source = {'buffer': audio_file.read(), 'mimetype': 'audio/wav'}
            options = PrerecordedOptions(model="nova-2", smart_format=True)
            response = deepgram.listen.prerecorded.v("1").transcribe_file(source, options)
            transcript = response.results.channels[0].alternatives[0].transcript
            print(f"User transcript: {transcript}")

        # If transcription is empty, do nothing
        if not transcript.strip():
            return history, "", None

        # Now, process the transcribed text through the chat interface
        async for history_update, text_update, audio_update in movie_chat_interface(transcript, history):
            # We yield the final state from the text processing
            final_history = history_update
            final_text = text_update
            final_audio = audio_update

        return final_history, "", final_audio # Clear textbox, return history and audio

    except Exception as e:
        print(f"Error in STT or chat processing: {e}")
        error_msg = f"Sorry, I couldn't process the audio. Error: {e}"
        history.append([None, error_msg])
        return history, "", None
    finally:
        # Clean up the temporary audio file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def movie_chat_interface(user_input: str, history: list):
    """
    This function processes the user's text input (either typed or from STT),
    gets a response from the chatbot, and generates TTS audio for the response.
    """
    history = history or []
    
    if not user_input.strip():
        yield history, "", None
        return

    if chatbot_instance is None or deepgram is None:
        history.append([user_input, "Error: Chatbot or Deepgram not initialized."])
        yield history, "", None
        return

    history.append([user_input, "Thinking..."])
    yield history, "", None

    tts_audio_path = None # Define path here to use in finally block
    try:
        response_text = await chatbot_instance.chat(user_input)
        history[-1][1] = response_text

        # --- UPDATED TTS LOGIC ---
        if response_text: # Only generate audio if there is a response
            if len(response_text) > 2000:
                print("Warning: Response text exceeds 2000 characters, may hit API limits.")
            
            speak_options = SpeakOptions(
                model="aura-2-juno-en",
                encoding="linear16",
                container="wav"
            )
            
            # 1. Call the streaming endpoint
            response = deepgram.speak.v("1").stream(
                {'text': response_text},
                speak_options
            )
            
            # 2. Get the synchronous BytesIO stream from the response
            audio_stream = response.stream

            # 3. Write the audio bytes to a temporary file for Gradio to play
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio_stream.read())
                tts_audio_path = tmpfile.name
        # --- END OF UPDATE ---
            
        yield history, "", tts_audio_path

    except Exception as e:
        error_msg = f"An error occurred: {e}"
        print(f"Error during chatbot response/TTS generation: {e}")
        history[-1][1] = error_msg
        yield history, "", None


# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft(), title="Entertainment Companion") as demo:
    gr.Markdown(
        """
        # 🎬 Entertainment Companion Chatbot 🍿
        Ask me anything about **Movies** and **TV Shows**, or ask for **Recommendations**!
        You can type your query or use the microphone to speak.
        """
    )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=400,
        avatar_images=("https://img.icons8.com/color/48/user.png", "https://img.icons8.com/fluency/48/chatbot.png"),
        show_copy_button=True,
    )

    # Audio component for TTS output (hidden, used for autoplay)
    tts_audio = gr.Audio(visible=False, autoplay=True)



    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter your query here, or use the microphone above...",
            container=False,
        )
        submit_btn = gr.Button("SUBMIT", variant="primary", scale=0, icon="https://img.icons8.com/fluency/24/filled-sent.png")
        audio_input = gr.Audio(sources=["microphone"],type="numpy", label="Speak", scale=1)

    with gr.Row():
         gr.Examples(
            examples=[
                "Recommend me a sci-fi movie from the 2020s",
                "What's the plot of Inception?",
                "Best horror movies of all time",
                "Shows like prison break"
            ],
            inputs=txt,
            label="Example Queries"
        )

    # Custom clear function
    def clear_chat():
        if chatbot_instance:
            chatbot_instance.memory.clear()
            print("Chat history cleared.")
        return [], "", None

    clear_btn = gr.Button("CLEAR", variant="secondary", scale=0, icon="https://img.icons8.com/fluency/24/delete-forever.png")


    # --- Component Connections ---

    # Typing text and submitting
    txt.submit(movie_chat_interface, [txt, chatbot], [chatbot, txt, tts_audio])
    submit_btn.click(movie_chat_interface, [txt, chatbot], [chatbot, txt, tts_audio])

    # Speaking and submitting
    audio_input.stop_recording(process_audio_and_chat, [audio_input, chatbot], [chatbot, txt, tts_audio])
    
    # Clearing the chat
    clear_btn.click(clear_chat, outputs=[chatbot, txt, tts_audio])


# --- Launch the Gradio App ---
if __name__ == "__main__":
    if chatbot_instance is None or deepgram is None:
        print("Application cannot start due to initialization errors.")
    else:
        demo.queue().launch(debug=True)