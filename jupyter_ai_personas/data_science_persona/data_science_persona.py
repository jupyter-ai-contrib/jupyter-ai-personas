"""
Simple Data Science Persona with notebook cell reading using Agno framework.
This persona demonstrates how to read notebook cells using the agno framework.
Enhanced with active notebook detection capabilities.
"""

import boto3
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from jupyter_ydoc import YNotebook
from agno.agent import Agent
from agno.models.aws import AwsBedrock
from langchain_core.messages import HumanMessage

# Import our notebook tools
from .ynotebook_wrapper import YNotebookToolsWrapper

session = boto3.Session()


class SimpleDataSciencePersona(BasePersona):
    """
    Simple Data Science Persona with notebook cell reading capabilities.
    This persona can read and analyze notebook cells using the agno framework.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize notebook tools
        self.notebook_tools = None
        
    @property
    def defaults(self):
        return PersonaDefaults(
            name="SimpleDataSciencePersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Simple data science persona with notebook cell reading capabilities.",
            system_prompt="I can read and analyze notebook cells to help you understand your data and code structure.",
        )

    def set_notebook_instance(self, ynotebook: YNotebook):
        """
        Set the YNotebook instance for notebook operations.
        
        Args:
            ynotebook: The YNotebook instance to use
        """
        self.notebook_tools = YNotebookToolsWrapper(ynotebook)
        
        # NEW: Try to get additional context from Jupyter AI if available
        # This would be set by Jupyter AI when creating the persona
        if hasattr(self, '_notebook_path'):
            self.notebook_tools.set_notebook_context(path=self._notebook_path)
        if hasattr(self, '_kernel_id'):
            self.notebook_tools.set_notebook_context(kernel_id=self._kernel_id)

    def initialize_notebook_agent(self):
        """Initialize the notebook reading agent"""
        model_id = self.config.lm_provider_params["model_id"]
        
        notebook_agent = Agent(
            name="notebook_reader",
            role="Notebook analyst who can read and analyze notebook cells",
            model=AwsBedrock(
                id=model_id,
                session=session
            ),
            instructions=[
                "You can read and analyze notebook cells to help users understand their data and code",
                "When asked to read a specific cell, provide the cell content and basic analysis",
                "When asked about notebook structure, provide overview information",
                "When asked to search cells, find relevant cells containing specified terms",
                "Be helpful and provide clear explanations of what you find in the notebook"
            ],
            markdown=True,
            show_tool_calls=True
        )
        
        return notebook_agent

    async def process_message(self, message: Message):
        """Process messages using simple notebook reading functionality"""
        message_text = message.body
        
        # Get chat history
        history = YChatHistory(ychat=self.ychat, k=2)
        messages = await history.aget_messages()
        
        history_text = ""
        if messages:
            history_text = "\nPrevious conversation:\n"
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"
        
        # Initialize the notebook agent
        notebook_agent = self.initialize_notebook_agent()
        
        # Create context with notebook information
        context = f"""User Request: {message_text}

Chat History: {history_text}

"""
        
        if self.notebook_tools:
            try:
                # NEW: Get active notebook information
                active_info = self.notebook_tools.get_active_notebook_info()
                
                # NEW: Use the enhanced summary method
                context += self.notebook_tools.get_notebook_summary()
                
                # NEW: Log active notebook detection for debugging
                if active_info['active']:
                    print(f"[SimpleDataSciencePersona] Active notebook: {active_info['path']}")
                    print(f"[SimpleDataSciencePersona] Detection source: {active_info['detection_source']}")
                
                # Handle specific commands (existing code remains the same)
                if "read cell" in message_text.lower():
                    # Extract cell number
                    words = message_text.split()
                    cell_num = None
                    for word in words:
                        if word.isdigit():
                            cell_num = int(word)
                            break
                    
                    if cell_num is not None:
                        cell_content = self.notebook_tools.read_cell(cell_num)
                        context += f"\nCell {cell_num} content:\n```\n{cell_content}\n```\n"
                
                elif "notebook info" in message_text.lower() or "show all cells" in message_text.lower():
                    info = self.notebook_tools.get_notebook_info()
                    context += f"\nNotebook has {info['cell_count']} cells (indexes 0-{info['max_index']})\n"
                    
                    # Add preview of cells
                    for i in range(min(info['cell_count'], 3)):
                        cell_content = self.notebook_tools.read_cell(i)
                        preview = cell_content[:100] + "..." if len(cell_content) > 100 else cell_content
                        context += f"Cell {i} preview: {preview}\n"
                
                elif "search" in message_text.lower():
                    # Extract search term
                    search_term = "pandas"  # Default
                    if "search for" in message_text.lower():
                        search_term = message_text.lower().split("search for", 1)[1].strip()
                    elif "find" in message_text.lower():
                        search_term = message_text.lower().split("find", 1)[1].strip().replace("cells with", "").strip()
                    
                    matching_cells = self.notebook_tools.search_cells(search_term)
                    context += f"\nSearch results for '{search_term}': Found in cells {matching_cells}\n"
                    
                    for cell_idx in matching_cells[:2]:  # Show first 2 matches
                        cell_content = self.notebook_tools.read_cell(cell_idx)
                        context += f"Cell {cell_idx}: {cell_content}\n"
                
                # NEW: Add command to show active notebook info
                elif "which notebook" in message_text.lower() or "current notebook" in message_text.lower():
                    if active_info['path']:
                        context += f"\nCurrently working with: {active_info['path']}\n"
                        context += f"Detection method: {active_info['detection_source']}\n"
                    else:
                        context += "\n❓ Unable to determine the current notebook path\n"
                    
            except Exception as e:
                context += f"❌ Error accessing notebook: {str(e)}\n"
        else:
            context += "❌ No notebook access available\n"
        
        # Get response from the agent
        response = notebook_agent.run(
            context,
            stream=False,
            stream_intermediate_steps=False,
            show_full_reasoning=True,
        )
        
        response_content = response.content
        
        async def response_iterator():
            yield response_content
        
        await self.stream_message(response_iterator())
    
    # NEW: Optional method to receive context from Jupyter AI
    def set_notebook_context(self, path: str = None, kernel_id: str = None):
        """
        Set notebook context information from Jupyter AI.
        
        Args:
            path: The notebook file path
            kernel_id: The kernel ID
        """
        if path:
            self._notebook_path = path
        if kernel_id:
            self._kernel_id = kernel_id
            
        # If notebook tools already initialized, update the context
        if self.notebook_tools:
            self.notebook_tools.set_notebook_context(path=path, kernel_id=kernel_id)