from telegram.ext import ContextTypes, ConversationHandler
from telegram import (
    Update,
    ReplyKeyboardRemove)

import time
import json
import html
# import openai - No longer needed directly for API calls here
from openai import BadRequestError, APITimeoutError # Import specific errors potentially raised
import asyncio
import traceback
from typing import Dict

from db.MySqlConn import config
from buttons import get_project_root
from config import (
    TYPING_REPLY,
    logger)


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the photos and asks for a location."""
    project_root = get_project_root()
    user = update.message.from_user
    if len(update.message.photo) != 0:
        await update.message.reply_text(text='Only text, thanks!')
        photo_file = await update.message.photo[-1].get_file()
        # can't get photo's name
        await photo_file.download_to_drive(
            f'{project_root}/data/photos/{user.name}-{time.strftime("%Y%m%d-%H%M%S")}.jpg')
        logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    else:
        await update.message.reply_text(text='Only text, thanks!')
        if update.message.document:
            file = await update.message.document.get_file()
            await file.download_to_drive(
                f'{project_root}/data/documents/{user.name}-{time.strftime("%Y%m%d-%H%M%S")}.jpg')
        if update.message.video:
            video = await update.message.video.get_file()
            await video.download_to_drive(
                f'{project_root}/data/videos/{user.name}-{time.strftime("%Y%m%d-%H%M%S")}.jpg')
    return TYPING_REPLY


def facts_to_str(user_data: Dict[str, str]) -> str:
    """Helper function for formatting the gathered user info."""
    facts = [f'{key} - {value}' for key, value in user_data.items()]
    return "\n".join(facts).join(['\n', '\n'])


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Conversation canceled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    # Log the error before we do anything else, so we can see it even if something breaks.
    logger.error("Exception while handling an update:", exc_info=context.error)

    # traceback.format_exception returns the usual python message about an exception, but as a
    # list of strings rather than a single string, so we have to join them together.
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)

    # Build the message with some markup and additional information about what happened.
    # You might need to add some logic to deal with messages longer than the 4096-character limit.
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    message = (
        f"An exception was raised while handling an update\n"
        f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
        "</pre>\n\n"
        f"<pre>error type = {type(context.error)}</pre>"
        f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
        f"<pre>prompt = {html.escape(str(update.message.text))}</pre>\n\n"
        f"<pre>{html.escape(tb_string)}</pre>"
    )

    # Finally, send the message
    error_reply = ""
    # Check for errors potentially raised by the openai library (used by OpenRouter client)
    # Note: OpenRouter might have its own content filtering policies, adjust message if needed.
    if isinstance(context.error, BadRequestError):
         # This could be due to various issues: invalid model, bad request format, potentially content filtering by OpenRouter.
         # The message might need refinement based on OpenRouter's specific error details if available.
        error_reply = f"API Request Error: {context.error}. This might be due to an invalid request, model issue, or content policy. Please check your input or configuration."
    elif isinstance(context.error, APITimeoutError) or isinstance(context.error, asyncio.TimeoutError):
        error_reply = "API request timed out. Please try again later."
    # Add other specific error checks if needed (e.g., AuthenticationError, RateLimitError from openai library)

    if error_reply:
        await update.message.reply_text(error_reply, parse_mode="Markdown", disable_web_page_preview=True)
    else:
        await update.message.reply_text(
            "Oops, our servers are overloaded due to high demand. Please take a break and try again later!",
            parse_mode="Markdown", disable_web_page_preview=True)
        await context.bot.send_message(
            chat_id=config["DEVELOPER_CHAT_ID"], text=message[:4096], parse_mode="HTML"
        )
