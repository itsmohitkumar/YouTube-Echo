import logging
import os
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_openai import ChatOpenAI
from src.helpers import (
    TranscriptException,
    TranscriptProcessor,
    ConfigManager,
    FileManager,
)
from src.youtube import (
    InvalidUrlException,
    NoTranscriptReceivedException,
    YouTubeTranscriptManager,
)

# Load environment variables from a .env file
load_dotenv()

# FastAPI app initialization
app = FastAPI()

# General error message for unexpected errors
GENERAL_ERROR_MESSAGE = "An unexpected error occurred. Please check the logs for more details."

class YoutubeEcho:
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=logging.INFO
        )
    
    def get_available_models(self, model_type: Literal["gpts", "embeddings"], api_key: str = "") -> List[str]:
        """
        Retrieve a filtered list of available model IDs from OpenAI's API or environment variables,
        based on the specified model type.
        """
        openai.api_key = api_key
        selectable_model_ids = list(ConfigManager.get_default_config_value(f"available_models.{model_type}"))

        available_model_ids = os.getenv("AVAILABLE_MODEL_IDS")
        if available_model_ids:
            return list(filter(lambda m: m in available_model_ids.split(","), selectable_model_ids))

        try:
            available_model_ids = [model.id for model in openai.models.list()]
            os.environ["AVAILABLE_MODEL_IDS"] = ",".join(available_model_ids)
            return list(filter(lambda m: m in available_model_ids, selectable_model_ids))
        except openai.AuthenticationError as e:
            logging.error("An authentication error occurred: %s", str(e))
        except Exception as e:
            logging.error("An unexpected error occurred: %s", str(e))

        return []

    def is_api_key_valid(self, api_key: str) -> bool:
        """Checks the validity of an OpenAI API key."""
        openai.api_key = api_key
        try:
            openai.models.list()  # Attempt to list models to validate the API key
            logging.info("API key validation successful")
            return True
        except openai.AuthenticationError as e:
            logging.error("An authentication error occurred: %s", str(e))
        except Exception as e:
            logging.error("An unexpected error occurred: %s", str(e))

        return False

    class SummarizeRequest(BaseModel):
        video_url: str
        custom_prompt: Optional[str] = None
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.9
        model: Optional[str] = "gpt-3.5-turbo"

    class SummarizeResponse(BaseModel):
        summary: str
        estimated_cost: float

    @app.post("/summarize", response_model=SummarizeResponse)
    async def summarize_video(request: SummarizeRequest):
        instance = YoutubeEcho()  # Create an instance of the YoutubeEcho class
        if not instance.is_api_key_valid(os.getenv("OPENAI_API_KEY")):
            raise HTTPException(status_code=400, detail="Invalid API Key")

        try:
            vid_metadata = YouTubeTranscriptManager.get_video_metadata(request.video_url)
            transcript = YouTubeTranscriptManager.fetch_youtube_transcript(request.video_url)

            cb = OpenAICallbackHandler()
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=request.temperature,
                model=request.model,
                top_p=request.top_p,
                callbacks=[cb],
                max_tokens=2048,
            )

            transcript_processor = TranscriptProcessor(llm)
            resp = transcript_processor.get_transcript_summary(
                transcript,
                custom_prompt=request.custom_prompt
            )

            return YoutubeEcho.SummarizeResponse(
                summary=resp,
                estimated_cost=cb.total_cost,
            )

        except (InvalidUrlException, NoTranscriptReceivedException, TranscriptException) as e:
            logging.error("An error occurred: %s", str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error("An unexpected error occurred: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=GENERAL_ERROR_MESSAGE)
    
# Run the application using Uvicorn
if __name__ == "__main__":
    app_instance = YoutubeEcho()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
