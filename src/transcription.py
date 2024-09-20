import whisper
import os
import logging
from pytubefix import YouTube
from typing import Optional
from src.youtube import YouTubeTranscriptManager

# Initialize logging
logging.basicConfig(level=logging.INFO)

class AudioDownloader:
    """Handles downloading of audio from YouTube videos."""

    def __init__(self, download_folder_path: str):
        self.download_folder_path = download_folder_path
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensures that the download directory exists."""
        os.makedirs(self.download_folder_path, exist_ok=True)

    def download_mp3(self, video_id: str) -> Optional[str]:
        """Downloads the audio of a YouTube video and saves it as an MP3 file.

        Args:
            video_id (str): The ID of the YouTube video.

        Returns:
            Optional[str]: The full path to the MP3 audio file, or None if the download fails.
        """
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            audio_metadata = YouTubeTranscriptManager.get_video_metadata(video_url)
            audio_filename = audio_metadata.get("name", f"{video_id}.mp3")
            audio_filepath = os.path.join(self.download_folder_path, audio_filename)

            yt = YouTube(url=video_url)
            stream = yt.streams.get_audio_only()
            if stream:
                stream.download(output_path=self.download_folder_path, filename=audio_filename)
                logging.info("Audio downloaded successfully: %s", audio_filepath)
                return audio_filepath
            else:
                logging.error("No audio stream found for video %s", video_id)
                return None
        except Exception as e:
            logging.error("Failed to download MP3 for video %s: %s", video_id, str(e))
            return None


class WhisperTranscriber:
    """Handles transcription of audio files using the Whisper model."""

    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)

    def generate_transcript(self, file_path: str) -> Optional[str]:
        """Transcribes the audio file at the given path using the Whisper model.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            Optional[str]: The transcription as plain text, or None if transcription fails.
        """
        try:
            transcription = self.model.transcribe(file_path)
            transcript_text = transcription.get("text", "")
            logging.info("Transcription successful for file: %s", file_path)
            return transcript_text
        except Exception as e:
            logging.error("Failed to transcribe file %s: %s", file_path, str(e))
            return None

# Example usage
if __name__ == "__main__":
    download_folder_path = "./downloads"
    video_id = "dQw4w9WgXcQ"  # Replace with your video ID

    # Initialize downloader and transcriber
    downloader = AudioDownloader(download_folder_path=download_folder_path)
    transcriber = WhisperTranscriber(model_name="base")

    # Download the audio
    audio_file_path = downloader.download_mp3(video_id=video_id)

    # Generate the transcript if the download was successful
    if audio_file_path:
        transcript = transcriber.generate_transcript(file_path=audio_file_path)
        if transcript:
            print("Transcript:\n", transcript)
        else:
            logging.error("Failed to generate transcript for %s", audio_file_path)
