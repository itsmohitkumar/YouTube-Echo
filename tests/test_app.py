import pytest
import os
from flask import json
from app import app, YoutubeEcho

# Set the testing environment and mock API keys
os.environ['FLASK_ENV'] = 'production'
os.environ['LANGCHAIN_API_KEY'] = 'test_langchain_api_key'  # Set mock keys
os.environ['OPENAI_API_KEY'] = 'test_openai_api_key'

@pytest.fixture
def client(mocker):
    """A test client for the app."""
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the index route returns a 200 status code."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<title>" in response.data  # Replace with a known static part of the HTML

def test_summarize_valid_request(client, mocker):
    """Test the /summarize route with valid input."""
    mocker.patch.object(YoutubeEcho, 'is_api_key_valid', return_value=True)
    mocker.patch.object(YoutubeEcho, 'summarize_video', return_value=("This is a summary.", 0.05))

    data = {
        'video_url': 'https://www.youtube.com/watch?v=sample_valid_video',
        'custom_prompt': 'Summarize this video.',
        'temperature': 0.7,
        'top_p': 0.9,
        'model': 'gpt-3.5-turbo'
    }

    response = client.post('/summarize', data=json.dumps(data), content_type='application/json')  # Add content_type
    response_json = json.loads(response.data)

    assert response.status_code == 200
    assert 'summary' in response_json

def test_summarize_invalid_api_key(client, mocker):
    """Test the /summarize route with an invalid API key."""
    mocker.patch.dict(os.environ, {
        "OPENAI_API_KEY": "invalid_api_key",
        "LANGCHAIN_API_KEY": "dummy_langchain_api_key"
    })
    
    data = {
        'video_url': 'https://www.youtube.com/watch?v=sample_valid_video',
        'custom_prompt': 'Summarize this video.',
        'temperature': 0.7,
        'top_p': 0.9,
        'model': 'gpt-3.5-turbo'
    }

    response = client.post('/summarize', data=json.dumps(data), content_type='application/json')  # Add content_type
    response_json = json.loads(response.data)

    assert response.status_code == 401
    assert 'error' in response_json
    assert response_json['error'] == 'Invalid or missing OpenAI API Key'

def test_ask_followup_valid_request(client, mocker):
    """Test the /ask_followup route with valid input."""
    mocker.patch.object(YoutubeEcho, 'ask_followup_question', return_value=("This is the follow-up response.", 0.02))

    data = {
        'followup_question': 'What is the main point of the video?'
    }

    response = client.post('/ask_followup', data=data)  # Change here

    response_json = json.loads(response.data)

    assert response.status_code == 200
    assert 'response' in response_json
    assert response_json['response'] == "This is the follow-up response."
    assert response_json['cost'] == 0.02

def test_ask_followup_error(client, mocker):
    """Test the /ask_followup route when there is an error processing the question."""
    mocker.patch.object(YoutubeEcho, 'ask_followup_question', return_value=(None, "Error processing the question."))

    data = {
        'followup_question': 'What is the main point of the video?'
    }

    response = client.post('/ask_followup', data=data)
    response_json = json.loads(response.data)

    assert response.status_code == 400  # Change to an appropriate status code for bad requests
    assert 'error' in response_json
    assert response_json['error'] == "Error processing the question."