from openai import AzureOpenAI
from db.MySqlConn import config
from ai import OPENAI_CHAT_COMPLETION_OPTIONS


class AzureAIClient:
    def __init__(self):
        self.open_ai_config = {
            'api_key': config["AI"]["TOKEN"],
            'azure_endpoint': config["AI"]["BASE"],
            'api_version': config["AI"]["VERSION"]
        }
        # Ensure the model is configured for Azure
        if "MODEL" not in config.get("AI", {}):
             raise ValueError("AI configuration in config.yaml is missing required key: MODEL (for Azure)")
        self.model = config["AI"]["MODEL"] # Add model attribute
        self.client = AzureOpenAI(**self.open_ai_config)

    def generate_image(self, prompt) -> str:
        response = self.client.images.generate(
            model="dalle3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = response.data[0].url
        if image_url:
            return image_url
        else:
            raise ValueError("Azure image generation response did not contain a valid URL.")

    def chat_completions(self, messages: list):
        answer = ""
        completion = self.client.chat.completions.create(
            model=OPENAI_CHAT_COMPLETION_OPTIONS["model"],
            messages=messages
        )
        # print(completion.choices[0].message)
        return completion
