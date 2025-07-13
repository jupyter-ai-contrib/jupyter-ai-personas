import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import asyncio
from dataclasses import dataclass

# Force pytest to load asyncio plugin
pytest_plugins = ("pytest_asyncio",)


# mock classes for base_persona
@dataclass
class PersonaDefaults:
    name: str = "MockPersona"
    avatar_path: str = "/mock/path"
    description: str = "Mock description"
    system_prompt: str = "Mock system prompt"


class BasePersona:
    def __init__(
        self,
        ychat=None,
        manager=None,
        config_manager=None,
        log=None,
        message_interrupted=None,
    ):
        self.ychat = ychat
        self.manager = manager
        self.config_manager = config_manager or MagicMock()
        self.log = log
        self.message_interrupted = message_interrupted
        self.name = "PRReviewPersona"

        # Set up config_manager with required attributes
        self.config_manager.lm_provider = MagicMock()
        self.config_manager.lm_provider.name = "test_provider"
        self.config_manager.lm_provider_params = {"model_id": "test_model"}

    @property
    def defaults(self):
        return PersonaDefaults()

    async def stream_message(self, message_iterator):
        async for message in message_iterator:
            pass

    def send_message(self, message):
        # Mock implementation for send_message
        pass


# mock YChatHistory class
class YChatHistory:
    def __init__(self, ychat=None, k=None):
        self.ychat = ychat
        self.k = k

    async def aget_messages(self):
        return []


# PersonaAwareness class
class PersonaAwareness:
    def __init__(self, *args, **kwargs):
        self.outdated_timeout = 30000
        self._heartbeat_task = None

    async def _start_heartbeat(self):
        return

    async def set_local_state(self, *args, **kwargs):
        return

    async def set_local_state_field(self, *args, **kwargs):
        return


# Mock asyncio operations
@pytest.fixture(autouse=True)
def mock_asyncio_operations():
    # Mock asyncio.to_thread to actually call the function and handle exceptions
    async def mock_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def mock_create_task(coro):
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        return mock_task

    with (
        patch("asyncio.to_thread", side_effect=mock_to_thread),
        patch("asyncio.create_task", side_effect=mock_create_task),
    ):
        yield


patch(
    "jupyter_ai.personas.persona_awareness.PersonaAwareness", PersonaAwareness
).start()
patch("jupyter_ai.personas.base_persona.BasePersona", BasePersona).start()
patch("jupyter_ai.personas.base_persona.PersonaDefaults", PersonaDefaults).start()
patch("jupyter_ai.history.YChatHistory", YChatHistory).start()
