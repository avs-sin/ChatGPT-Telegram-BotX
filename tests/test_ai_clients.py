import unittest
from unittest.mock import patch, MagicMock
from ai.openai import OpenAIClient
from ai.openrouter import OpenRouterClient
from db.MySqlConn import config


class TestOpenAIClient(unittest.TestCase):
    @patch('ai.openai.OpenAI')
    def setUp(self, mock_openai):
        self.mock_client = mock_openai.return_value
        self.client = OpenAIClient()

    def test_chat_completions(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"role": "assistant", "content": "Test response"})]
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.client.chat_completions([{"role": "user", "content": "Test"}])
        self.assertEqual(result.choices[0].message["content"], "Test response")

    def test_generate_image(self):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="http://test.url")]
        self.mock_client.images.generate.return_value = mock_response

        result = self.client.generate_image("test prompt")
        self.assertEqual(result, "http://test.url")
        self.mock_client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="test prompt",
            size="1024x1024",
            quality="standard",
            n=1
        )


class TestOpenRouterClient(unittest.TestCase):
    @patch('ai.openrouter.OpenAI')
    def setUp(self, mock_openai):
        self.mock_client = mock_openai.return_value
        self.client = OpenRouterClient()

    def test_chat_completions(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"role": "assistant", "content": "Test response"})]
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.client.chat_completions([{"role": "user", "content": "Test"}])
        self.assertEqual(result.choices[0].message["content"], "Test response")
        # Verify OpenRouter-specific headers/base_url are set
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}]
        )

    def test_generate_image(self):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="http://test.url")]
        self.mock_client.images.generate.return_value = mock_response

        result = self.client.generate_image("test prompt")
        self.assertEqual(result, "http://test.url")
        self.mock_client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="test prompt",
            size="1024x1024",
            quality="standard",
            n=1
        )


if __name__ == '__main__':
    unittest.main()
