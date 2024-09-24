import os
from fastapi.testclient import TestClient
import pytest
from dotenv import load_dotenv
from app import app  # Import your FastAPI app

# Load environment variables from the .env file
load_dotenv()

# Fixture for setting up the TestClient
@pytest.fixture
def client():
    """Create a TestClient for testing the FastAPI app."""
    return TestClient(app)

def test_summarize_video(client):
    """Test summarizing a video with a valid URL and API key."""

    # Ensure the API Key is set in the environment from .env
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "API key not found. Please set it in the .env file."

    # Define the payload for the POST request
    payload = {
        "video_url": "https://www.youtube.com/watch?v=A8t2NOERe5U",  # Replace with your video URL
        "custom_prompt": "Please summarize this video.",
        "temperature": 0.7,
        "top_p": 0.9,
        "model": "gpt-3.5-turbo"
    }

    # Send a POST request to the /summarize endpoint
    response = client.post("/summarize", json=payload)

    # Log the response content for debugging (optional)
    # You can comment this out if you don't need the output during normal runs
    print(response.json())  

    # Check that the response status code is 200 (OK)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}. Response: {response.json()}"
