import json
import logging
import requests
from requests.exceptions import RequestException
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    Transcript,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.formatters import TextFormatter
from .helpers import YouTubeUtils, LanguagePreferences, FileManager

OEMBED_PROVIDER = "https://noembed.com/embed"

class NoTranscriptReceivedException(Exception):
    """Custom exception for when no transcript is received."""

    def __init__(self, url: str):
        self.message = "Unfortunately, no transcript was found for this video. Therefore, a summary can't be provided :slightly_frowning_face:"
        self.url = url
        super().__init__(self.message)

    def log_error(self):
        """Logs the error with additional context."""
        logging.error("Could not find a transcript for %s", self.url)


class InvalidUrlException(Exception):
    """Custom exception for invalid URLs."""

    def __init__(self, message: str, url: str):
        self.message = message
        self.url = url
        super().__init__(self.message)

    def log_error(self):
        """Logs the error with additional context."""
        logging.error("Invalid URL provided: %s", self.url)


class YouTubeTranscriptManager:
    """Handles fetching and analyzing YouTube transcripts."""

    @staticmethod
    def get_video_metadata(url: str):
        """Fetches metadata for a YouTube video."""

        if not ("youtube.com" in url or "youtu.be" in url):
            raise InvalidUrlException(
                "Seems not to be a YouTube URL :confused: If you are convinced that it's a YouTube URL, report the bug.",
                url,
            )

        try:
            response = requests.get(OEMBED_PROVIDER, params={"url": url}, timeout=5)
            response.raise_for_status()
        except RequestException as e:
            logging.warning("Can't retrieve metadata for provided video URL: %s", str(e))
            return None

        json_response = response.json()
        FileManager.save_response_as_file(
            dir_name="./video_meta",
            filename=f"{json_response['title']}",
            file_content=json_response,
            content_type="json",
        )
        return {
            "name": json_response["title"],
            "channel": json_response["author_name"],
            "provider_name": json_response["provider_name"],
        }

    @staticmethod
    def fetch_youtube_transcript(url: str) -> str:
        """Fetches the transcript of a YouTube video."""

        video_id = YouTubeUtils.extract_youtube_video_id(url)
        if video_id is None:
            raise InvalidUrlException("Something is wrong with the URL :confused:", url)

        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=LanguagePreferences.get_preffered_languages()
            )
        except CouldNotRetrieveTranscript as e:
            logging.error("Failed to retrieve transcript for URL: %s", str(e))
            raise NoTranscriptReceivedException(url)

        formatter = TextFormatter()
        return formatter.format_transcript(transcript)

    @staticmethod
    def analyze_transcripts(video_id: str):
        """Analyzes the available transcripts for a YouTube video."""
        try:
            transcript_list: list[Transcript] = YouTubeTranscriptApi.list_transcripts(
                video_id
            )
        except Exception as e:
            logging.error("An error occurred when fetching transcripts: %s", str(e))
            return

        for t in transcript_list:
            if t.is_generated:
                logging.info(
                    f"Found auto-generated transcript in {t.language} ({t.language_code})!"
                )
            else:
                logging.info(
                    f"Found manual transcript in {t.language} ({t.language_code})!"
                )


# Example usage
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual YouTube video URL

    # Fetch video metadata
    metadata = YouTubeTranscriptManager.get_video_metadata(youtube_url)
    if metadata:
        print("Video Metadata:", metadata)

    # Fetch transcript
    try:
        transcript_text = YouTubeTranscriptManager.fetch_youtube_transcript(youtube_url)
        print("Transcript:", transcript_text)
    except NoTranscriptReceivedException as e:
        e.log_error()
    except InvalidUrlException as e:
        e.log_error()
