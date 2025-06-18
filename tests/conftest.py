import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from dataclasses import dataclass

# mock classes for base_persona
@dataclass
class PersonaDefaults:
    name: str = "MockPersona"
    avatar_path: str = "/mock/path"
    description: str = "Mock description"
    system_prompt: str = "Mock system prompt"

class BasePersona:
    def __init__(self, ychat=None, manager=None, config=None, log=None, message_interrupted=None):
        self.ychat = ychat
        self.manager = manager
        self.config = config
        self.log = log
        self.message_interrupted = message_interrupted
        self.name = "PRReviewPersona" 
    
    @property
    def defaults(self):
        return PersonaDefaults()
    
    async def stream_message(self, message_iterator):
        async for message in message_iterator:
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

patch('jupyter_ai.personas.persona_awareness.PersonaAwareness', PersonaAwareness).start()
patch('jupyter_ai.personas.base_persona.BasePersona', BasePersona).start()
patch('jupyter_ai.personas.base_persona.PersonaDefaults', PersonaDefaults).start()
patch('jupyter_ai.history.YChatHistory', YChatHistory).start()