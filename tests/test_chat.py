import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.app.chat import TranscriptProcessor

class TestTranscriptProcessor(unittest.TestCase):

    def setUp(self):
        # Create a mock LLM
        self.mock_llm = MagicMock()
        # Initialize the TranscriptProcessor with the mock LLM
        self.processor = TranscriptProcessor(self.mock_llm)

    def test_process_transcript(self):
        # Prepare test data
        excerpts = [
            Document(page_content="This is the first excerpt."),
            Document(page_content="This is the second excerpt."),
        ]
        
        # Simulate LLM responses
        self.mock_llm.generate.return_value = MagicMock(
            generations=[
                [MagicMock(text="First excerpt corrected.")],
                [MagicMock(text="Second excerpt corrected.")]
            ]
        )
        
        # Call the process_transcript method
        result = self.processor.process_transcript(excerpts)
        
        # Expected result
        expected_result = "First excerpt corrected.\n\nSecond excerpt corrected."
        
        # Verify the output
        self.assertEqual(result, expected_result)

    def test_process_transcript_error_handling(self):
        # Prepare test data
        excerpts = [
            Document(page_content="This will cause an error."),
        ]
        
        # Make the LLM raise an exception
        self.mock_llm.generate.side_effect = Exception("LLM processing error")
        
        # Assert that processing raises an error
        with self.assertRaises(Exception) as context:
            self.processor.process_transcript(excerpts)
        
        # Verify the exception message
        self.assertEqual(str(context.exception), "LLM processing error")

if __name__ == '__main__':
    unittest.main()
