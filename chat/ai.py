from typing import Tuple, AsyncGenerator, Optional
from openai.types.chat import ChatCompletionChunk # Import the chunk type
from config import token
from db.MySqlConn import config
from ai.openai import OpenAIClient
from ai.azure import AzureAIClient
from ai.openrouter import OpenRouterClient # Added import
from ai import OPENAI_CHAT_COMPLETION_OPTIONS


def init_client():
    ai_type = config["AI"].get("TYPE", "openai").lower() # Default to openai if not specified
    if ai_type == "azure":
        client = AzureAIClient()
    elif ai_type == "openrouter":
        client = OpenRouterClient()
    elif ai_type == "openai": # Explicitly check for openai
        client = OpenAIClient()
    else:
        raise ValueError(f"Unsupported AI TYPE configured: {ai_type}")
    return client


async def ChatCompletionsAI(logged_in_user, messages) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
    """Yields the generated answer incrementally and the final finish reason."""
    level = logged_in_user.get("level")

    ai = init_client()
    answer = ""
    # Use the model configured in the specific client instance
    model_to_use = ai.model if hasattr(ai, 'model') else OPENAI_CHAT_COMPLETION_OPTIONS["model"]
    # Pass other options like temperature if available in config or client
    # chat_options = {**OPENAI_CHAT_COMPLETION_OPTIONS} # Start with defaults
    # chat_options['model'] = model_to_use # Override model
    # if hasattr(ai, 'temperature'): chat_options['temperature'] = ai.temperature # Example

    with ai.client.chat.completions.with_streaming_response.create(
            messages=messages,
            max_tokens=token[level],
            model=model_to_use, # Use the determined model
            # Pass other relevant options if needed, e.g., temperature
            # **chat_options
            ) as response:
        # Removed explicit type hint for r as it caused issues
        for r in response.parse():
            # Assuming r has .choices attribute at runtime based on openai library behavior
            if r.choices:
                delta = r.choices[0].delta
                if delta.content:
                    answer += delta.content
                yield answer, r.choices[0].finish_reason


async def GenerateImage(prompt):
    ai = init_client()
    image_url = ai.generate_image(prompt)
    return image_url
