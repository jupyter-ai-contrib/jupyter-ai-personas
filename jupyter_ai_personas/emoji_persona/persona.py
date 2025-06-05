from typing import Any

import emoji
from jupyterlab_chat.models import Message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from jupyter_ai.history import YChatHistory
from jupyter_ai.personas import BasePersona, PersonaDefaults
from jupyter_ai.personas.jupyternaut.prompt_template import JupyternautVariables

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


_SYSTEM_PROMPT_FORMAT = """
<instructions>

You are {{persona_name}}, an AI agent provided in JupyterLab through the 'Jupyter AI' extension.

Jupyter AI is an installable software package listed on PyPI and Conda Forge as `jupyter-ai`.

When installed, Jupyter AI adds a chat experience in JupyterLab that allows multiple users to collaborate with one or more agents like yourself.

You are not a language model, but rather an AI agent powered by a foundation model `{{model_id}}`, provided by '{{provider_name}}'.

You are receiving a request from a user in JupyterLab. Your goal is to respond to user's query with emojis (:emoji: format) in response.

You will receive any provided context and a relevant portion of the chat history.

The user's request is located at the last message. Please fulfill the user's request to the best of your ability.
</instructions>

<context>
{% if context %}The user has shared the following context:

{{context}}
{% else %}The user did not share any additional context.{% endif %}
</context>
""".strip()

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            _SYSTEM_PROMPT_FORMAT, template_format="jinja2"
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

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
            description="The emoji agent, that responds with emojis.",
            system_prompt="...",
        )

    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]

        runnable = self.build_runnable()
        variables = JupyternautVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
        )

        variables_dict = variables.model_dump()
        reply = runnable.invoke(variables_dict)
        print(f"reply from model: {reply}")
        reply = emoji.emojize(reply, variant="emoji_type")
        print(f"reply after emojize: {reply}")
        self.send_message(reply)

    def build_runnable(self) -> Any:
        llm = self.config.lm_provider(**self.config.lm_provider_params)
        
        runnable = PROMPT_TEMPLATE | llm | StrOutputParser()
        runnable = RunnableWithMessageHistory(
            runnable=runnable,  #  type:ignore[arg-type]
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=0),
            input_messages_key="input",
            history_messages_key="history",
        )
        return runnable