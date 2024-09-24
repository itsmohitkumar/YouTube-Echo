from flask import Flask, render_template, request, jsonify
import logging
import os
import json
import openai
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_openai import ChatOpenAI
from src.app.helpers import (
    TranscriptException,
    TranscriptProcessor,
)
from src.app.youtube import (
    InvalidUrlException,
    NoTranscriptReceivedException,
    YouTubeTranscriptManager,
)

# Load environment variables
load_dotenv()

# Load configuration from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

# Wrap OpenAI client with LangSmith tracing
client = wrap_openai(openai)

app = Flask(__name__)

GENERAL_ERROR_MESSAGE = "An unexpected error occurred. Please check the logs for more details."

class YoutubeEcho:
    instance = None  # Class variable for singleton instance

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(YoutubeEcho, cls).__new__(cls)
            cls.instance.setup_logging()
            cls.instance.summary = None
            cls.instance.json_data = None  # Variable to store JSON data
        return cls.instance

    def setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=logging.INFO
        )

    def is_api_key_valid(self, api_key: str) -> bool:
        """Check the validity of an OpenAI API key."""
        openai.api_key = api_key
        try:
            openai.models.list()
            logging.info("API key validation successful")
            return True
        except openai.AuthenticationError as e:
            logging.error("Authentication error: %s", str(e))
        except Exception as e:
            logging.error("Unexpected error: %s", str(e))
        return False

    @traceable  # Auto-trace this method
    def summarize_video(self, video_url, custom_prompt=None, temperature=None, top_p=None, model=None):
        if temperature is None:
            temperature = config.get("temperature", 1.0)  # Use config value or default to 1.0
        if top_p is None:
            top_p = config.get("top_p", 1.0)  # Use config value or default to 1.0
        if model is None:
            model = config["default_model"]["gpt"]  # Use default model from config

        try:
            vid_metadata = YouTubeTranscriptManager.get_video_metadata(video_url)
            transcript = YouTubeTranscriptManager.fetch_youtube_transcript(video_url)

            self.json_data = vid_metadata  # Store JSON data
            cb = OpenAICallbackHandler()
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                model=model,
                top_p=top_p,
                callbacks=[cb],
                max_tokens=2048,
            )

            transcript_processor = TranscriptProcessor(llm)
            self.summary = transcript_processor.get_transcript_summary(transcript, custom_prompt=custom_prompt)
            return self.summary, cb.total_cost  # Return summary and cost

        except (InvalidUrlException, NoTranscriptReceivedException, TranscriptException) as e:
            logging.error("Error: %s", str(e))
            return None, str(e)
        except Exception as e:
            logging.error("Unexpected error: %s", str(e), exc_info=True)
            return None, GENERAL_ERROR_MESSAGE

    @traceable  # Auto-trace this method
    def ask_followup_question(self, followup_question: str):
        """Process the follow-up question using stored summary and JSON data."""
        try:
            cb = OpenAICallbackHandler()
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=config.get("temperature", 1.0),
                model=config["default_model"]["gpt"],
                callbacks=[cb],
                max_tokens=2048,
            )

            # Use both summary and JSON data for context
            full_prompt = (
                f"Based on the following summary of a video:\n\n"
                f"{self.summary}\n\n"
                f"Here is some relevant information from the video:\n\n"
                f"{self.json_data}\n\n"
                f"Please answer this follow-up question: {followup_question}"
            )
            
            logging.info("Asking follow-up question with prompt: %s", full_prompt)

            # Use the `invoke` method instead of `__call__`
            response = llm.invoke(full_prompt)

            # Convert the response to a string
            response_text = getattr(response, 'content', str(response))

            return response_text, cb.total_cost

        except Exception as e:
            logging.error("Error processing follow-up question: %s", str(e))
            return None, "Error processing the question."

@app.route('/')
def index():
    """Render the homepage with a form for video URL and prompt."""
    app_title = config["app_title"]
    available_gpt_models = config["available_models"]["gpts"]
    available_embeddings = config["available_models"]["embeddings"]
    return render_template(
        'index.html',
        app_title=app_title,
        available_gpt_models=available_gpt_models,
        available_embeddings=available_embeddings,
        config_data=config  # Pass the entire config data as config_data
    )

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle the summarization form submission."""
    video_url = request.form.get('video_url')
    custom_prompt = request.form.get('custom_prompt')
    temperature = float(request.form.get('temperature', config.get("temperature", 1.0)))
    top_p = float(request.form.get('top_p', config.get("top_p", 1.0)))
    model = request.form.get('model', config["default_model"]["gpt"])

    youtube_echo = YoutubeEcho()
    if not youtube_echo.is_api_key_valid(os.getenv("OPENAI_API_KEY")):
        return jsonify({'error': 'Invalid API Key'})

    summary, cost = youtube_echo.summarize_video(video_url, custom_prompt, temperature, top_p, model)

    if summary:
        return jsonify({'summary': summary, 'cost': cost})  # Return summary and cost
    else:
        return jsonify({'error': cost})

@app.route('/ask_followup', methods=['POST'])
def ask_followup():
    """Handle follow-up question submissions."""
    followup_question = request.form.get('followup_question')

    youtube_echo = YoutubeEcho()
    response, cost = youtube_echo.ask_followup_question(followup_question)

    if response:
        return jsonify({'response': response, 'cost': cost})
    else:
        return jsonify({'error': cost})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
