from typing import List
from langchain.chat_models.base import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessage

# System message that sets the task for the LLM
SYSTEM_PROMPT = (
    "You are going to receive excerpts from an automatically generated video transcript. "
    "Your task is to convert every excerpt into structured text. Ensure that the content "
    "of the excerpts remains unchanged. Add appropriate punctuation, correct any grammatical "
    "errors, remove filler words, and divide the text into logical paragraphs, separating them "
    "with a single new line. The final output should be in plain text and only include the modified "
    "transcript excerpt without any prelude."
)

# User message prompt template that will format each transcript excerpt
user_prompt = HumanMessagePromptTemplate.from_template(
    """Here is part {number} from the original transcript, delimited by ---

    ---
    {transcript_excerpt}
    ---
    """
)

def process_transcript(transcript_excerpts: List[Document], llm: BaseChatModel) -> str:
    batch_messages = []
    
    # Create batch messages for the LLM
    for num, excerpt in enumerate(transcript_excerpts, start=1):
        batch_messages.append(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                user_prompt.format(number=num, transcript_excerpt=excerpt.page_content),
            ]
        )
    
    # Generate the response using the LLM
    response = llm.generate(batch_messages)
    
    # Join and return the generated responses as a single string
    return "\n\n".join(gen[0].text for gen in response.generations)
