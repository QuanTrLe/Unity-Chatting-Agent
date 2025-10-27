### FOR USING WITH UI ####

import asyncio
import traceback
import os
import uuid  # <-- FIX: Added the missing import

from collections.abc import AsyncIterator
from pprint import pformat

import gradio as gr

from routing_agent import (
    get_initialized_routing_agent as routing_agent,
)
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.cloud import speech, texttospeech


# Create a temporary directory for audio files
if not os.path.exists("temp_audio"):
    os.makedirs("temp_audio")
    
    
APP_NAME = 'routing_app'
USER_ID = 'default_user'
SESSION_ID = 'default_session'

SESSION_SERVICE = InMemorySessionService()
# Create a runner from the routing agent we imported
ROUTING_AGENT_RUNNER = Runner(
    agent=routing_agent().create_agent(),
    app_name=APP_NAME,
    session_service=SESSION_SERVICE,
)


async def transcribe_audio(file_path: str) -> str:
    """Transcribes audio file to text using Google Cloud Speech-to-Text."""
    try:
        client = speech.SpeechAsyncClient()

        with open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US",
            sample_rate_hertz=48000,
        )

        print("Waiting for transcription to complete...")
        response = await client.recognize(config=config, audio=audio)
        
        if response.results and response.results[0].alternatives:
            return response.results[0].alternatives[0].transcript
        else:
            return "Could not transcribe audio."
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Sorry, I had trouble understanding the audio."
    

async def generate_speech(text: str) -> str:
    """Converts text to speech and saves it as an MP3 file."""
    try:
        client = texttospeech.TextToSpeechAsyncClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = await client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        file_name = f"temp_audio/{uuid.uuid4()}.mp3"
        with open(file_name, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{file_name}"')
        
        return file_name
    except Exception as e:
        print(f"Error during speech generation: {e}")
        return ""
        
        
async def main():
    """Main gradio app."""
    with gr.Blocks(
        theme=gr.themes.Ocean(), title='A2A Host Agent with Audio'
    ) as demo:
        
        # FIX 1: Add a unique ID to the chatbot and set the type
        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", type="messages")

        # FIX 2: Add this gr.HTML component containing the JavaScript
        # This script will find the last audio element in the chatbot and play it.
        gr.HTML("""
            <script>
                // Function to play the last audio element in the chatbot
                function playLastAudio() {
                    const chatbot = document.getElementById('chatbot');
                    if (chatbot) {
                        const audioElements = chatbot.querySelectorAll('audio');
                        if (audioElements.length > 0) {
                            const lastAudio = audioElements[audioElements.length - 1];
                            // Check if the audio is not already playing to avoid errors
                            if (lastAudio.paused) {
                                lastAudio.play().catch(e => console.error("Autoplay failed:", e));
                            }
                        }
                    }
                }

                // Use a MutationObserver to watch for changes in the chatbot
                const observer = new MutationObserver((mutationsList, observer) => {
                    for(const mutation of mutationsList) {
                        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                            // When new messages are added, check for and play audio
                            playLastAudio();
                        }
                    }
                });

                // Start observing the chatbot for configured mutations
                // We need to wait for the chatbot element to be in the DOM
                const interval = setInterval(() => {
                    const chatbot = document.getElementById('chatbot');
                    if (chatbot) {
                        observer.observe(chatbot, { childList: true, subtree: true });
                        clearInterval(interval); // Stop checking once the chatbot is found
                    }
                }, 100); // Check every 100ms
            </script>
        """)
        
        with gr.Row():
            text_box = gr.Textbox(placeholder="Type your message here...", scale=3)
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Or record your voice", scale=1)
        
        submit_btn = gr.Button("Submit", variant="primary")

        # The 'respond' function remains exactly the same as the last version
        async def respond(audio, text, chat_history):
            # ... (no changes needed in this function)
            user_message = ""
            # --- Handle User Input ---
            if audio:
                user_message = await transcribe_audio(audio)
                if user_message and user_message != "Could not transcribe audio.":
                    chat_history.append({"role": "user", "content": f"🎤: *{user_message}*"})
                else: 
                    chat_history.append({"role": "assistant", "content": "I'm sorry, I couldn't understand the audio. Please try again."})
                    yield chat_history, None, ""
                    return
            elif text:
                user_message = text
                chat_history.append({"role": "user", "content": f"👤: {user_message}"})
            else:
                yield chat_history, None, ""
                return

            yield chat_history, None, ""

            session = await SESSION_SERVICE.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
            )
            if session is None:
                print(f"Session '{SESSION_ID}' not found. Creating a new one.")
                await SESSION_SERVICE.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
                )

            # --- Get AI Response ---
            final_response_text = ""
            try:
                event_iterator: AsyncIterator[Event] = ROUTING_AGENT_RUNNER.run_async(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    new_message=types.Content(
                        role='user', parts=[types.Part(text=user_message)]
                    ),
                )

                async for event in event_iterator:
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            final_response_text = ''.join(
                                [p.text for p in event.content.parts if p.text]
                            )
                        break
            except Exception as e:
                print(f"Error getting response from agent: {e}")
                final_response_text = "Sorry, an error occurred while I was thinking."
            
            chat_history.append({"role": "assistant", "content": f"🔊: {final_response_text}"})
            yield chat_history, None, ""

            # --- Generate and add audio ---
            if final_response_text:
                audio_file_path = await generate_speech(final_response_text)
                if audio_file_path and os.path.exists(audio_file_path) and os.path.getsize(audio_file_path) > 0:
                    print(f"Adding audio file to chat: {audio_file_path}")
                    chat_history.append({"role": "assistant", "content": (audio_file_path,)})
                    yield chat_history, None, ""
                else:
                    print("Failed to generate or find a valid audio file.")


        submit_btn.click(
            fn=respond,
            inputs=[audio_input, text_box, chatbot],
            outputs=[chatbot, audio_input, text_box],
        )

    print('Launching Gradio interface...')
    demo.queue().launch(
        server_name='localhost',
        server_port=10004,
        share=True,
    )
    print('Gradio application has been shut down.')


if __name__ == '__main__':
    asyncio.run(main())