from flask import Flask, render_template, request, jsonify
import logging
import os
import json
import openai
from dotenv import load_dotenv
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

# Set LangChain tracing and API key from the config file
os.environ["LANGCHAIN_PROJECT"] = config["langchain"]["project_name"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables.")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Define the project endpoint from config
LANGCHAIN_ENDPOINT = config["langchain"]["endpoint"]

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
        logging.debug(f"Validating API Key: {api_key}")  # Debug log for API key
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
    def summarize_video(self, video_url, custom_prompt=None, temperature=None, top_p=None, model=None, api_key=None):
        # Get API key from either function argument or environment
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            logging.error("OpenAI API Key is not set. Please check your environment variables or input.")
            return None, "OpenAI API Key is not set."

        # Check OpenAI API Key validity
        if not self.is_api_key_valid(openai_api_key):
            logging.error("Invalid or missing OpenAI API Key.")
            return None, 'Invalid or missing OpenAI API Key'

        # Get LangChain API key from environment
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        if not langchain_api_key:
            logging.error("Invalid or missing LangChain API Key.")
            return None, 'Invalid or missing LangChain API Key'

        # Set default values for temperature, top_p, and model
        temperature = temperature or config.get("temperature", 1.0)  # Use config value or default to 1.0
        top_p = top_p or config.get("top_p", 1.0)  # Use config value or default to 1.0
        model = model or config["default_model"]["gpt"]  # Use default model from config

        # Validate the video_url
        if not video_url:
            logging.error("Invalid video URL: %s", video_url)
            return None, "Invalid video URL."  # Early return for invalid URL

        try:
            # Fetch metadata and transcript for the provided video URL
            vid_metadata = YouTubeTranscriptManager.get_video_metadata(video_url)
            transcript = YouTubeTranscriptManager.fetch_youtube_transcript(video_url)

            # Store JSON data for later use
            self.json_data = vid_metadata

            # Initialize OpenAI callback and language model
            cb = OpenAICallbackHandler()
            llm = ChatOpenAI(
                api_key=openai_api_key,  # Use the passed or environment key
                temperature=temperature,
                model=model,
                top_p=top_p,
                callbacks=[cb],
                max_tokens=2048,
            )

            # Process the transcript to generate a summary
            transcript_processor = TranscriptProcessor(llm)
            self.summary = transcript_processor.get_transcript_summary(transcript, custom_prompt=custom_prompt)
            return self.summary, cb.total_cost  # Return the summary and the cost

        except KeyError as e:
            logging.error("KeyError: %s", str(e))
            return None, "Error retrieving video metadata: missing required information."
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
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logging.error("OpenAI API Key is not set. Please check your environment variables.")
                return None, "OpenAI API Key is not set."

            cb = OpenAICallbackHandler()
            llm = ChatOpenAI(
                api_key=openai_api_key,
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

    # Check if API key exists in environment variables
    api_key_exists = os.getenv("OPENAI_API_KEY") is not None

    return render_template(
        'index.html',
        app_title=app_title,
        available_gpt_models=available_gpt_models,
        available_embeddings=available_embeddings,
        config_data=config,  # Pass the entire config data as config_data
        api_key_exists=api_key_exists  # Pass API key existence status
    )

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle the summarization form submission."""
    video_url = request.form.get('video_url')
    custom_prompt = request.form.get('custom_prompt')
    temperature = float(request.form.get('temperature', config.get("temperature", 1.0)))
    top_p = float(request.form.get('top_p', config.get("top_p", 1.0)))
    model = request.form.get('model', config["default_model"]["gpt"])

    # Get the API key from the form if provided
    api_key = request.form.get('api_key')

    # Log the received API key
    if api_key:
        logging.info(f"Received API Key.")
    else:
        logging.info("No API Key provided from form; falling back to environment.")

    youtube_echo = YoutubeEcho()

    # Pass the API key (if any) to the summarization method
    summary, cost = youtube_echo.summarize_video(video_url, custom_prompt, temperature, top_p, model, api_key)

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
    app.run(debug=False, host='0.0.0.0', port=5000)  # Set debug to False for production
