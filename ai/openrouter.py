from openai import OpenAI, BadRequestError, APITimeoutError # Import relevant errors
from db.MySqlConn import config
from ai import OPENAI_CHAT_COMPLETION_OPTIONS # Re-use existing options for now

# Note: OpenRouter uses an OpenAI-compatible API structure.
# We can reuse the openai library by pointing it to the OpenRouter endpoint.

class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API using an OpenAI-compatible interface.
    Reads configuration (API key, base URL, model) from the global 'config' object.
    """
    def __init__(self):
        # Ensure required AI config keys are present
        if not all(k in config.get("AI", {}) for k in ["TOKEN", "BASE", "MODEL"]):
             raise ValueError("AI configuration in config.yaml is missing required keys: TOKEN, BASE, MODEL")

        self.open_router_config = {
            'api_key': config["AI"]["TOKEN"],
            'base_url': config["AI"]["BASE"] # Expecting OpenRouter base URL here
        }
        # Optional: Add custom headers required by OpenRouter for identification
        # See: https://openrouter.ai/docs#headers
        headers = {
             "HTTP-Referer": config.get("APP_URL", "http://localhost"), # Get from config or default
             "X-Title": config.get("APP_NAME", "ChatGPT-Telegram-BotX"), # Get from config or default
        }
        self.client = OpenAI(**self.open_router_config, default_headers=headers)
        # Use the specific model from config
        self.model = config["AI"]["MODEL"]

    def generate_image(self, prompt) -> str:
        """
        Generates an image using an OpenRouter-compatible endpoint.
        Requires appropriate model configuration in config.yaml (e.g., AI.IMAGE_MODEL).
        """
        # Use a specific image model identifier from config or a default.
        # Check OpenRouter documentation for available image models and their identifiers.
        image_model_identifier = config["AI"].get("IMAGE_MODEL")
        if not image_model_identifier:
            # Log or raise error if image model isn't configured
            print("Error: AI.IMAGE_MODEL not configured in config.yaml for image generation.")
            raise NotImplementedError("Image generation requires AI.IMAGE_MODEL to be set in the configuration.")

        try:
            print(f"Generating image with model: {image_model_identifier}") # Debug log
            response = self.client.images.generate(
                model=image_model_identifier,
                prompt=prompt,
                size="1024x1024",  # Common size, adjust if needed for the specific model
                quality="standard", # Common quality, adjust if needed
                n=1
            )
            if response.data and len(response.data) > 0 and response.data[0].url:
                image_url = response.data[0].url
                print(f"Image generated successfully: {image_url}") # Debug log
                return image_url
            else:
                 print("Error: Image generation response did not contain a valid URL.") # Debug log
                 raise ValueError("Image generation response did not contain a valid URL.")
        except BadRequestError as e:
            # Handle specific errors like invalid requests (e.g., model not found, bad prompt)
            print(f"Error generating image via OpenRouter ({image_model_identifier}): Invalid request - {e}")
            raise ValueError(f"Failed to generate image due to an invalid request (check model or prompt): {e}")
        except Exception as e:
            # Log the error appropriately using the project's logger if available
            # logger.error(f"Error generating image via OpenRouter: {e}")
            print(f"Error generating image via OpenRouter ({image_model_identifier}): {e}")
            # Raise a more specific error or return a user-friendly message
            raise RuntimeError(f"Image generation failed for OpenRouter model '{image_model_identifier}'. Error: {e}")


    def chat_completions(self, messages: list):
        """
        Performs a non-streaming chat completion using the configured OpenRouter model.
        Note: The main streaming logic is handled externally (e.g., in chat/ai.py).
        This method is primarily for testing or non-streaming use cases.
        """
        try:
            print(f"Performing chat completion with model: {self.model}") # Debug log
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # Add other parameters like temperature, top_p etc. if needed from config or defaults
                # temperature=config["AI"].get("TEMPERATURE", 0.7),
            )
            print("Chat completion successful.") # Debug log
            return completion
        except APITimeoutError as e:
            print(f"Error during chat completion via OpenRouter ({self.model}): Timeout - {e}")
            raise TimeoutError(f"OpenRouter API request timed out: {e}")
        except BadRequestError as e:
            print(f"Error during chat completion via OpenRouter ({self.model}): Invalid request - {e}")
            raise ValueError(f"OpenRouter API request failed (check model or messages): {e}")
        except Exception as e:
            # Log the error
            print(f"Error during chat completion via OpenRouter ({self.model}): {e}")
            # Re-raise the exception or handle it as appropriate
            raise RuntimeError(f"Chat completion failed for OpenRouter model '{self.model}'. Error: {e}")
