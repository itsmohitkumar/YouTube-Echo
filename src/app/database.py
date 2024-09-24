import logging
from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    UUIDField,
)

# Initialize the SQLite database connection
SQL_DB = SqliteDatabase("data/videos.sqlite3")

class BaseModel(Model):
    """Base model class for all database models, ensuring they share the same database connection."""
    class Meta:
        database = SQL_DB


class Video(BaseModel):
    """Model representing a YouTube video in the database."""

    yt_video_id = CharField(unique=True)  # Unique identifier for the YouTube video
    title = CharField()  # Title of the video
    link = CharField()  # Link to the video
    channel = CharField(null=True)  # YouTube channel name (optional)
    saved_on = DateTimeField(null=True)  # Timestamp for when the video was saved

    def chroma_collection_id(self):
        """
        Retrieves the ID of the associated Chroma collection for this video.

        Returns:
            str or None: The ID of the Chroma collection, or None if not found.
        """
        try:
            transcript = Transcript.get(Transcript.video == self)
            return transcript.chroma_collection_id
        except Transcript.DoesNotExist:
            logging.warning("Transcript for video %s does not exist.", self.yt_video_id)
            return None

    def chroma_collection_name(self):
        """
        Retrieves the name of the associated Chroma collection for this video.

        Returns:
            str or None: The name of the Chroma collection, or None if not found.
        """
        try:
            transcript = Transcript.get(Transcript.video == self)
            return transcript.chroma_collection_name
        except Transcript.DoesNotExist:
            logging.warning("Transcript for video %s does not exist.", self.yt_video_id)
            return None


class Transcript(BaseModel):
    """Model representing transcripts for YouTube videos in the database."""

    video = ForeignKeyField(Video, backref="transcripts")  # Foreign key to the Video model
    language = CharField(null=True)  # Language of the transcript (optional)
    preprocessed = BooleanField(null=True)  # Indicates if the transcript has been preprocessed
    chunk_size = IntegerField(null=True)  # Size of chunks used for the processed transcript
    original_token_num = IntegerField(null=True)  # Number of tokens in the original transcript
    processed_token_num = IntegerField(null=True)  # Number of tokens in the processed transcript
    chroma_collection_id = UUIDField(null=True)  # ID of the associated Chroma collection
    chroma_collection_name = CharField(null=True)  # Name of the associated Chroma collection


def delete_video(video_title: str):
    """
    Deletes a video and its associated transcripts from the SQLite database.

    Args:
        video_title (str): The title of the video to be deleted.
    """
    try:
        # Retrieve the video object based on its title
        video = Video.get(Video.title == video_title)
        
        # Delete associated transcripts
        transcripts = Transcript.select().where(Transcript.video == video)
        if transcripts.exists():
            Transcript.delete().where(Transcript.video == video).execute()
            logging.info("Removed transcripts for video %s from SQLite.", video.yt_video_id)

        # Delete the video
        Video.delete_by_id(video.id)
        logging.info("Removed video %s from SQLite.", video.yt_video_id)

    except Video.DoesNotExist:
        logging.error("Video with title %s does not exist.", video_title)
    except Exception as e:
        logging.error("An error occurred when deleting entries from SQLite for video %s: %s", video_title, str(e))
