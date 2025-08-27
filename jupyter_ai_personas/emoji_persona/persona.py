from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from litellm import acompletion

import emoji


class EmojiPersona(BasePersona):
    """
    The Emoji persona, responds to your queries with emojis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="EmojiPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="The emoji agent, always responds with emojis.",
            system_prompt="You are an AI that responds to all user queries using only emojis. Do not use any words, only emojis. If you don't know how to respond with emojis, respond with a shrug emoji.",
        )

    async def process_message(self, message: Message):
        if not self.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return
        
        model_id = self.config_manager.chat_model
        model_args = self.config_manager.chat_model_args
        message.body = message.body + "\n\nDecorate your reply with emojis."
        response_aiter = await acompletion(
            **model_args,
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": message.body,
                },
            ],
            stream=False,
        )
        response_text = response_aiter.choices[0].message.content
        emoji_text = emoji.emojize(response_text, variant="emoji_type")
        self.send_message(emoji_text)