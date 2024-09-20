import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .helpers import num_tokens_from_string

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
