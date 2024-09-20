import logging
from typing import List
from langchain_core.documents import Document
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessage

# Configure logging to track the flow of the application and assist in debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TranscriptProcessor:
    """
    TranscriptProcessor: A class designed to handle the processing of transcript excerpts using a Language Learning Model (LLM).
    
    This class encapsulates the logic for transforming raw transcript excerpts into structured, readable text,
    ensuring that the output is grammatically correct, well-punctuated, and logically segmented.
    """
    
    # System-level prompt that provides the LLM with specific instructions on how to process the transcript
    SYSTEM_PROMPT = (
        "You are going to receive excerpts from an automatically generated video transcript. "
        "Your task is to convert every excerpt into structured text. Ensure that the content "
        "of the excerpts remains unchanged. Add appropriate punctuation, correct any grammatical "
        "errors, remove filler words, and divide the text into logical paragraphs, separating them "
        "with a single new line. The final output should be in plain text and only include the modified "
        "transcript excerpt without any prelude."
    )
    
    def __init__(self, llm: BaseChatModel):
        """
        Initializes the TranscriptProcessor with a specific Language Learning Model (LLM).
        
        :param llm: An instance of a BaseChatModel, responsible for generating the refined transcript.
        """
        self.llm = llm
        
        # Template to structure the user's prompt for each transcript excerpt
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            """Here is part {number} from the original transcript, delimited by ---
    
            ---
            {transcript_excerpt}
            ---
            """
        )
        logging.info("TranscriptProcessor initialized with the specified LLM.")

    def _create_batch_messages(self, transcript_excerpts: List[Document]) -> List[List[SystemMessage]]:
        """
        Creates a batch of system and user messages for the LLM to process.
        
        :param transcript_excerpts: A list of Document objects, each containing a portion of the transcript.
        :return: A list of message batches, where each batch includes the system prompt and the corresponding user prompt.
        """
        batch_messages = []
        
        # Enumerating through each transcript excerpt to create individual batches
        for num, excerpt in enumerate(transcript_excerpts, start=1):
            batch_messages.append(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    self.user_prompt.format(number=num, transcript_excerpt=excerpt.page_content),
                ]
            )
        
        logging.info(f"Created {len(batch_messages)} batch message(s) for processing.")
        return batch_messages

    def process_transcript(self, transcript_excerpts: List[Document]) -> str:
        """
        Processes a list of transcript excerpts by sending them through the LLM and compiling the results.
        
        :param transcript_excerpts: A list of Document objects representing the raw transcript excerpts.
        :return: A single string containing the refined transcript with appropriate structure and corrections.
        
        This method includes error handling to catch and log any issues that arise during processing.
        """
        try:
            # Generate batch messages from the given transcript excerpts
            batch_messages = self._create_batch_messages(transcript_excerpts)
            logging.info("Batch messages successfully created, initiating LLM processing.")
            
            # Send the batch messages to the LLM and generate a response
            response = self.llm.generate(batch_messages)
            logging.info("LLM has successfully generated the responses.")
            
            # Join the individual responses into a single, cohesive transcript
            refined_transcript = "\n\n".join(gen[0].text for gen in response.generations)
            logging.info("Transcript processing completed.")
            
            return refined_transcript
        
        except Exception as e:
            # Log the exception details before re-raising to provide insight into any failures
            logging.error(f"An error occurred while processing the transcript: {e}")
            raise

# Example usage:
# llm = YourLLMModel()  # Replace with an instance of your LLM model
# processor = TranscriptProcessor(llm)
# transcript_excerpts = [Document(page_content="..."), ...]
# result = processor.process_transcript(transcript_excerpts)
