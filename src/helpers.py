import json
import logging
import os
import re
import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Constants for the system prompt
SYSTEM_PROMPT = """You are an expert in processing video transcripts according to the user's request. 
For example, this could include summarization, question answering, or providing key insights.
"""

# Context windows for different models
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": {"total": 16385, "output": 4096},
    "gpt-4": {"total": 8192, "output": 4096},
    "gpt-4-turbo": {"total": 128000, "output": 4096},
    "gpt-4o": {"total": 128000, "output": 4096},
    "gpt-4o-mini": {"total": 128000, "output": 16000},
}

class ConfigManager:
    """Manages configuration file operations."""

    @staticmethod
    def get_default_config_value(key_path: str, config_file_path: str = "./config.json") -> str:
        """
        Retrieves a configuration value from a JSON file using a specified key path.

        Args:
            key_path (str): The key path in dot notation to retrieve the value.
            config_file_path (str): The path to the configuration file.

        Returns:
            str: The value associated with the specified key path.
        """
        with open(config_file_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

            keys = key_path.split(".")
            value = config
            for key in keys:
                value = value[key]  # Navigate through each level

            return value


class FileManager:
    """Handles file operations like saving content to a file."""

    @staticmethod
    def save_response_as_file(dir_name: str, filename: str, file_content, content_type: str = "text"):
        """
        Saves given content to a file in the specified directory, formatted as either plain text, JSON, or Markdown.

        Args:
            dir_name (str): The directory where the file will be saved.
            filename (str): The name of the file without extension.
            file_content: The content to be saved.
            content_type (str): The type of content: "text", "json", or "markdown". Defaults to "text".
        """
        filename = filename.replace("/", "_").replace("\\", "_")
        os.makedirs(dir_name, exist_ok=True)

        extensions = {"text": ".txt", "json": ".json", "markdown": ".md"}
        file_extension = extensions.get(content_type, ".txt")
        filename += file_extension

        file_path = os.path.join(dir_name, filename)

        with open(file_path, "w", encoding="utf-8") as file:
            if content_type == "json":
                json.dump(file_content, file, indent=4)
            else:
                file.write(file_content)

        logging.info("File saved at: %s", file_path)


class YouTubeUtils:
    """Contains utilities for extracting video IDs from YouTube URLs."""

    @staticmethod
    def extract_youtube_video_id(url: str):
        """
        Extracts the video ID from a given YouTube URL.

        Args:
            url (str): The YouTube URL from which the video ID is to be extracted.

        Returns:
            str or None: The extracted video ID as a string if found, otherwise None.
        """
        pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None


class TokenCounter:
    """Manages token counting based on the model's tokenization."""

    @staticmethod
    def num_tokens_from_string(string: str, model: str = "gpt-4o-mini") -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): The string to count tokens in.
            model (str): Name of the model. Default is 'gpt-4o-mini'.
        """
        encoding_name = tiktoken.encoding_name_for_model(model_name=model)
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))


class EnvironmentChecker:
    """Checks environment-related configurations."""

    @staticmethod
    def is_environment_prod() -> bool:
        """Checks if the environment is set to production."""
        return os.getenv("ENVIRONMENT") == "production"


class LanguagePreferences:
    """Handles preferred language settings."""

    @staticmethod
    def get_preffered_languages():
        """
        Retrieves the preferred languages for the system.

        Returns:
            list: A list of preferred language codes.
        """
        return ["en-US", "en", "de"]


class TranscriptProcessor:
    """Manages transcript processing operations, including summarization."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("user", "{input}"),
            ]
        )

    def get_transcript_summary(self, transcript_text: str, **kwargs) -> str:
        """
        Generates a summary from a video transcript using a language model.

        Args:
            transcript_text (str): The full transcript text of the video.
            **kwargs: Optional keyword arguments.
                - custom_prompt (str): A custom prompt to replace the default summary request.

        Raises:
            TranscriptException: If the transcript exceeds the model's context window.

        Returns:
            str: The summary or answer in markdown format.
        """
        user_prompt = f"""Based on the provided transcript of the video, create a summary that accurately captures the main topics and arguments. The summary should be in whole sentences and contain no more than 300 words.
            Additionally, extract key insights from the video to contribute to better understanding, emphasizing the main points and providing actionable advice.
            Here is the transcript, delimited by ---
            ---
            {transcript_text}
            ---
            Answer in markdown format strictly adhering to this schema:

            ## <short title for the video, consisting of maximum five words>

            <your summary>

            ## Key insights

            <unnumbered list of key insights>
            """

        if "custom_prompt" in kwargs:
            user_prompt = f"""{kwargs['custom_prompt']}
                Here is the transcript, delimited by ---
                ---
                {transcript_text}
                ---
                """

        transcript_tokens = TokenCounter.num_tokens_from_string(string=transcript_text, model=self.llm.model_name)
        prompt_tokens = TokenCounter.num_tokens_from_string(string=user_prompt, model=self.llm.model_name)
        context_window = CONTEXT_WINDOWS[self.llm.model_name]["total"]

        if transcript_tokens + prompt_tokens > context_window:
            exception_message = (
                f"Your transcript exceeds the context window of the chosen model ({self.llm.model_name}), which is {context_window} tokens. "
                "Consider the following options:\n"
                "1. Choose another model with a larger context window (such as gpt-4o).\n"
                "2. Use the 'Chat' feature to ask specific questions about the video, where there won't be a token limit.\n\n"
                "You can get more information on context windows for different models in the [official OpenAI documentation about models](https://platform.openai.com/docs/models)."
            )
            raise TranscriptException(message=exception_message, model_name=self.llm.model_name)

        chain = self.prompt | self.llm | StrOutputParser()
        logging.info("Generating transcript summary...")
        return chain.invoke({"input": user_prompt})


class TranscriptException(Exception):
    """Exception raised when the transcript length exceeds the context window of the language model."""

    def __init__(self, message: str, model_name: str):
        super().__init__(message)
        self.model_name = model_name

    def log_error(self):
        """Logs the error message with detailed exception information."""
        logging.error("Transcript too long for %s: %s", self.model_name, self.args[0], exc_info=True)
