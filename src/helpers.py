import json
import logging
import os
import re
import tiktoken
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are an expert in processing video transcripts according to the user's request. 
For example, this could include summarization, question answering, or providing key insights.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
    ]
)

# Information about OpenAI's GPT context windows
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": {"total": 16385, "output": 4096},
    "gpt-4": {"total": 8192, "output": 4096},
    "gpt-4-turbo": {"total": 128000, "output": 4096},
    "gpt-4o": {"total": 128000, "output": 4096},
    "gpt-4o-mini": {"total": 128000, "output": 16000},
}


class TranscriptTooLongForModelException(Exception):
    """Raised when the transcript length exceeds the context window of the language model."""

    def __init__(self, message: str, model_name: str):
        super().__init__(message)
        self.model_name = model_name

    def log_error(self):
        """Log error message with detailed exception information."""
        logging.error("Transcript too long for %s: %s", self.model_name, self.args[0], exc_info=True)


def get_transcript_summary(transcript_text: str, llm: ChatOpenAI, **kwargs) -> str:
    """
    Generates a summary from a video transcript using a language model.

    Args:
        transcript_text (str): The full transcript text of the video.
        llm (ChatOpenAI): The language model instance used for generating the summary.
        **kwargs: Optional keyword arguments.
            - custom_prompt (str): A custom prompt to replace the default summary request.

    Raises:
        TranscriptTooLongForModelException: If the transcript exceeds the model's context window.

    Returns:
        str: The summary or answer in markdown format.
    """

    # Define the default prompt for summarizing the transcript
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

    # Override with a custom prompt if provided
    if "custom_prompt" in kwargs:
        user_prompt = f"""{kwargs['custom_prompt']}
            Here is the transcript, delimited by ---
            ---
            {transcript_text}
            ---
            """

    # Calculate the number of tokens in the transcript and the prompt
    transcript_tokens = num_tokens_from_string(string=transcript_text, model=llm.model_name)
    prompt_tokens = num_tokens_from_string(string=user_prompt, model=llm.model_name)
    context_window = CONTEXT_WINDOWS[llm.model_name]["total"]

    # Check if the combined token count exceeds the model's context window
    if transcript_tokens + prompt_tokens > context_window:
        exception_message = (
            f"Your transcript exceeds the context window of the chosen model ({llm.model_name}), which is {context_window} tokens. "
            "Consider the following options:\n"
            "1. Choose another model with a larger context window (such as gpt-4o).\n"
            "2. Use the 'Chat' feature to ask specific questions about the video, where there won't be a token limit.\n\n"
            "You can get more information on context windows for different models in the [official OpenAI documentation about models](https://platform.openai.com/docs/models)."
        )
        raise TranscriptTooLongForModelException(message=exception_message, model_name=llm.model_name)

    # Create the summary using the language model
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_prompt})

def get_default_config_value(
    key_path: str,
    config_file_path: str = "./config.json",
) -> str:
    """
    Retrieves a configuration value from a JSON file using a specified key path.

    Args:
        config_file_path (str): A string representing the relative path to the JSON config file.

        key_path (str): A string representing the path to the desired value within the nested JSON structure,
                        with each level separated by a '.' (e.g., "level1.level2.key").

    Returns:
        The value corresponding to the key path within the configuration file. If the key path does not exist,
        a KeyError is raised.

    Raises:
        KeyError: If the specified key path is not found in the configuration.
    """
    with open(config_file_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

        keys = key_path.split(".")
        value = config
        for key in keys:
            value = value[key]  # Navigate through each level

        return value


def extract_youtube_video_id(url: str):
    """
    Extracts the video ID from a given YouTube URL.

    Args:
        url (str): The YouTube URL from which the video ID is to be extracted.

    Returns:
        str or None: The extracted video ID as a string if the URL is valid and the video ID is found, otherwise None.
    """
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def save_response_as_file(
    dir_name: str, filename: str, file_content, content_type: str = "text"
):
    """
    Saves given content to a file in the specified directory, formatted as either plain text, JSON, or Markdown.

    Args:
        dir_name (str): The directory where the file will be saved.
        filename (str): The name of the file without extension.
        file_content: The content to be saved. Can be a string for text or Markdown, or a dictionary/list for JSON.
        content_type (str): The type of content: "text" for plain text, "json" for JSON format, or "markdown" for Markdown format. Defaults to "text".
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


def get_preffered_languages():
    return ["en-US", "en", "de"]


def num_tokens_from_string(string: str, model: str = "gpt-4o-mini") -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string (str): The string to count tokens in.
        model (str): Name of the model. Default is 'gpt-4o-mini'
    """
    encoding_name = tiktoken.encoding_name_for_model(model_name=model)
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def is_environment_prod():
    if os.getenv("ENVIRONMENT") == "production":
        return True
    return False
