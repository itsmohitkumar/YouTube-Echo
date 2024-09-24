import unittest
from unittest.mock import patch, mock_open
from src.app.helpers import ConfigManager, FileManager, YouTubeUtils

class TestConfigManager(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_get_default_config_value(self, mock_file):
        value = ConfigManager.get_default_config_value("key")
        self.assertEqual(value, "value")
        mock_file.assert_called_once_with("./config.json", "r", encoding="utf-8")

class TestFileManager(unittest.TestCase):

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_response_as_file(self, mock_file, mock_makedirs):
        FileManager.save_response_as_file("test_dir", "test_file", "Test content", "text")
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_file().write.assert_called_once_with("Test content")

class TestYouTubeUtils(unittest.TestCase):

    def test_extract_youtube_video_id(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = YouTubeUtils.extract_youtube_video_id(url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")

if __name__ == "__main__":
    unittest.main()
