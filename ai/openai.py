from openai import OpenAI
from db.MySqlConn import config
from ai import OPENAI_CHAT_COMPLETION_OPTIONS


class OpenAIClient:
    def __init__(self):
        self.open_ai_config = {'api_key': config["AI"]["TOKEN"]}
        self.client = OpenAI(**self.open_ai_config)
        # Add model attribute, using the default from options
        self.model = OPENAI_CHAT_COMPLETION_OPTIONS.get("model", "gpt-3.5-turbo") # Use .get for safety

    def generate_image(self, prompt) -> str:
        response = self.client.images.generate(
            model="dall-e-3", # Keep using specific model for OpenAI DALL-E
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = response.data[0].url
        if image_url:
            return image_url
        else:
            raise ValueError("OpenAI image generation response did not contain a valid URL.")

    # For testing purposes
    def chat_completions(self, messages: list):
        answer = ""
        completion = self.client.chat.completions.create(
            model=OPENAI_CHAT_COMPLETION_OPTIONS["model"],
            messages=messages
        )
        # print(completion.choices[0].message)
        return completion
