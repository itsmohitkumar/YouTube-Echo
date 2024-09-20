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
    class Meta:
        database = SQL_DB


class Video(BaseModel):
    """Model for YouTube videos. Represents a table in a relational SQL database."""

    yt_video_id = CharField(unique=True)  # ID of the YouTube video (not a PK but uniquely identifies a video)
    title = CharField()
    link = CharField()
    channel = CharField(null=True)
    saved_on = DateTimeField(null=True)

    def chroma_collection_id(self):
        """Returns the ID of the associated Chroma collection."""
        try:
            transcript = Transcript.get(Transcript.video == self)
            return transcript.chroma_collection_id
        except Transcript.DoesNotExist:
            logging.warning("Transcript for video %s does not exist.", self.yt_video_id)
            return None

    def chroma_collection_name(self):
        """Returns the name of the associated Chroma collection."""
        try:
            transcript = Transcript.get(Transcript.video == self)
            return transcript.chroma_collection_name
        except Transcript.DoesNotExist:
            logging.warning("Transcript for video %s does not exist.", self.yt_video_id)
            return None


class Transcript(BaseModel):
    """Model for transcripts of the YouTube videos. Represents a table in a relational SQL database."""

    video = ForeignKeyField(Video, backref="transcripts")
    language = CharField(null=True)  # Language of the transcript
    preprocessed = BooleanField(null=True)  # Whether the transcript was preprocessed
    chunk_size = IntegerField(null=True)  # Chunk size used to split the (processed) transcript
    original_token_num = IntegerField(null=True)  # Number of tokens in the original transcript
    processed_token_num = IntegerField(null=True)  # Number of tokens in the processed transcript
    chroma_collection_id = UUIDField(null=True)  # ID of the associated collection in Chroma
    chroma_collection_name = CharField(null=True)  # Name of the associated collection in Chroma


def delete_video(video_title: str):
    """Deletes the video and associated transcripts from SQLite."""
    try:
        video = Video.get(Video.title == video_title)
        transcripts = Transcript.select().where(Transcript.video == video)
        if transcripts.exists():
            Transcript.delete().where(Transcript.video == video).execute()
            logging.info("Removed transcripts for video %s from SQLite.", video.yt_video_id)
        Video.delete_by_id(video.id)
        logging.info("Removed video %s from SQLite.", video.yt_video_id)
    except Video.DoesNotExist:
        logging.error("Video with title %s does not exist.", video_title)
    except Exception as e:
        logging.error("An error occurred when deleting entries from SQLite for video %s: %s", video_title, str(e))
