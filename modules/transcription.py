import whisper
import os
import logging
from modules.youtube import get_video_metadata
from pytubefix import YouTube
from typing import Optional

# Load the Whisper model
model = whisper.load_model("base")


def download_mp3(video_id: str, download_folder_path: str) -> Optional[str]:
    """Downloads the audio of a YouTube video and saves it as an MP3 file at the specified location.

    Args:
        video_id (str): The ID of the YouTube video.
        download_folder_path (str): The path to the folder where the MP3 file will be saved.

    Returns:
        Optional[str]: The full path to the MP3 audio file, or None if the download fails.
    """
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        audio_metadata = get_video_metadata(video_url)
        audio_filename = audio_metadata.get("name", "audio.mp3")
        audio_filepath = os.path.join(download_folder_path, audio_filename)

        # Ensure the directory exists
        os.makedirs(download_folder_path, exist_ok=True)

        yt = YouTube(url=video_url)
        stream = yt.streams.get_audio_only()
        if stream:
            stream.download(mp3=True, filename=audio_filepath)
            return audio_filepath
        else:
            logging.error("No audio stream found for video %s", video_id)
            return None
    except Exception as e:
        logging.error("Failed to download MP3 for video %s: %s", video_id, str(e))
        return None


def generate_transcript(file_path: str) -> Optional[str]:
    """Transcribes the audio file at the given path using the Whisper base model.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        Optional[str]: The transcription as plain text, or None if transcription fails.
    """
    try:
        transcription = model.transcribe(file_path)
        return transcription.get("text", "")
    except Exception as e:
        logging.error("Failed to transcribe file %s: %s", file_path, str(e))
        return None
