import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from langchain_core.messages import HumanMessage

from agno.agent import Agent
from agno.models.aws import AwsBedrock
from jupyter_ai_tools import ReadFileTool, WriteFileTool, ListDirectoryTool

# Import our ynotebook wrapper
from .ynotebook_wrapper import YNotebookToolsWrapper

logger = logging.getLogger(__name__)


class ContextCuratorPersona(BasePersona, ABC):
    """
    Context-aware base class for knowledge curator personas.
    Can be extended for different knowledge sources and domains.
    """
    
    def __init__(self, ynotebook=None, *args, **kwargs):
        """
        Initialize the knowledge curator.
        
        Args:
            ynotebook: Optional YNotebook instance for cell access
            *args, **kwargs: Additional arguments for BasePersona
        """
        super().__init__(*args, **kwargs)
        self.knowledge_dir = os.path.join(os.getcwd(), "knowledge")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        # Initialize ynotebook tools if available
        self.ynotebook_tools = YNotebookToolsWrapper(ynotebook)

    @property
    @abstractmethod
    def knowledge_source_config(self):
        """
        Define the knowledge source configuration.
        Must be implemented by subclasses.
        
        Returns:
            dict: Configuration with keys like 'base_url', 'source_name', 'domain'
        """
        pass

    @property
    def defaults(self):
        config = self.knowledge_source_config
        return PersonaDefaults(
            name="ContextCuratorPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description=f"Knowledge curator for {config['source_name']} - {config['description']}",
            system_prompt=f"I am a knowledge curator specialized in {config['domain']}. "
                         f"I analyze your questions and code to find relevant information "
                         f"from {config['source_name']}, then create curated knowledge files."
        )

    def get_current_cell_content(self) -> str:
        """
        Get current cell content using ynotebook tools if available.
        
        Returns:
            str: Current cell source code or empty string
        """
        try:
            if self.ynotebook_tools:
                return self.ynotebook_tools.read_current_cell()
            return ""
        except Exception as e:
            logger.warning(f"Could not get current cell content: {e}")
            return ""

    def analyze_notebook_context(self, user_input: str) -> dict:
        """
        Analyze the notebook context to provide richer information to the agent.
        
        Args:
            user_input: The user's input message
            
        Returns:
            dict: Context information about the notebook
        """
        context = {
            'current_cell': '',
            'notebook_info': {},
            'related_cells': [],
            'cell_count': 0
        }
        
        if not self.ynotebook_tools:
            return context
            
        try:
            # Get current cell
            context['current_cell'] = self.get_current_cell_content()
            
            # Get notebook info
            context['notebook_info'] = self.ynotebook_tools.get_notebook_info()
            context['cell_count'] = context['notebook_info'].get('cell_count', 0)
            
            # Search for related cells based on user input
            search_terms = user_input.lower().split()
            for term in search_terms:
                if len(term) > 3:  # Only search for meaningful terms
                    related = self.ynotebook_tools.search_cells(term)
                    context['related_cells'].extend(related)
            
            # Remove duplicates and limit results
            context['related_cells'] = list(set(context['related_cells']))[:5]
            
        except Exception as e:
            logger.warning(f"Error analyzing notebook context: {e}")
            
        return context

    def create_generic_agent_instructions(self, user_input: str, notebook_context: dict, config: dict) -> list:
        """
        Create generic agent instructions that work for any knowledge source.
        
        Args:
            user_input: User's input message
            notebook_context: Context from notebook analysis
            config: Knowledge source configuration
            
        Returns:
            list: Agent instructions
        """
        return [
            f"You are a knowledge curator for {config['source_name']}.",
            f"",
            f"CONTEXT:",
            f"User Input: {user_input}",
            f"Current Cell: {notebook_context['current_cell'] or 'No current cell content'}",
            f"Notebook has {notebook_context['cell_count']} cells",
            f"Related cells found: {notebook_context['related_cells'] if notebook_context['related_cells'] else 'None'}",
            f"Knowledge Source: {config['base_url']}",
            f"Domain: {config['domain']}",
            f"Knowledge Directory: {self.knowledge_dir}",
            f"",
            f"TASK: Create a comprehensive knowledge file based on the user's needs.",
            f"",
            f"STEP 1 - ANALYZE USER NEEDS:",
            f"Based on the user input and current cell content, determine:",
            f"- What specific topic or concept they need help with",
            f"- What level of detail is appropriate (beginner/intermediate/advanced)",
            f"- Whether they need conceptual explanation, practical examples, or both",
            f"",
            f"STEP 2 - FETCH RELEVANT CONTENT:",
            f"Retrieve relevant content from {config['source_name']}:",
            f"Base URL: {config['base_url']}",
            f"- Use HTTP requests to fetch content from the knowledge source",
            f"- Focus on content that directly addresses the user's question",
            f"- Look for both explanatory text and practical code examples",
            f"",
            f"STEP 3 - EXTRACT AND SYNTHESIZE:",
            f"From the fetched content, extract:",
            f"- Key concepts and explanations",
            f"- Practical code examples that work",
            f"- Common use cases and applications",
            f"- Best practices and potential pitfalls",
            f"",
            f"STEP 4 - CREATE STRUCTURED MARKDOWN:",
            f"Generate a well-organized markdown file with this structure:",
            f"",
            f"```markdown",
            f"# {{Main Topic}}",
            f"",
            f"## Overview",
            f"Brief explanation of what this topic is and why it's useful.",
            f"",
            f"## Key Concepts",
            f"- Important concept 1",
            f"- Important concept 2", 
            f"- Core terminology",
            f"",
            f"## Practical Examples",
            f"",
            f"### Basic Usage",
            f"```python",
            f"# Simple, working example",
            f"import pandas as pd",
            f"# ... relevant code",
            f"```",
            f"",
            f"### Advanced Patterns",
            f"```python",
            f"# More sophisticated example",
            f"# ... advanced usage",
            f"```",
            f"",
            f"## Common Use Cases",
            f"- When to use this approach",
            f"- Typical scenarios where this is helpful",
            f"- Real-world applications",
            f"",
            f"## Best Practices",
            f"- Recommended approaches",
            f"- Performance considerations",
            f"- Common mistakes to avoid",
            f"",
            f"## Related Topics",
            f"- Other concepts that build on this",
            f"- Prerequisites for understanding this topic",
            f"```",
            f"",
            f"STEP 5 - SAVE KNOWLEDGE FILE:",
            f"- Create a descriptive filename (e.g., 'pandas_groupby_operations.md')",
            f"- Save to: {self.knowledge_dir}",
            f"- Use WriteFileTool to save the markdown content",
            f"",
            f"REQUIREMENTS:",
            f"- Focus specifically on what the user asked about",
            f"- Include working, tested code examples",
            f"- Keep explanations clear and practical",
            f"- Make the content immediately actionable",
            f"- Ensure all code examples are complete and runnable",
            f"",
            f"After completing the task, provide a summary that includes:",
            f"- What topic was covered",
            f"- The filename and location of the created knowledge file",
            f"- A brief description of what the user can expect to find in the file"
        ]

    def create_curator_agent(self, user_input: str, notebook_context: dict) -> Agent:
        """
        Create a curator agent for the specific knowledge source.
        
        Args:
            user_input: User's input message
            notebook_context: Context from notebook analysis
            
        Returns:
            Agent: Configured Agno agent
        """
        if not hasattr(self.config, 'lm_provider_params') or 'model_id' not in self.config.lm_provider_params:
            raise ValueError("Model ID not found in configuration")

        model_id = self.config.lm_provider_params["model_id"]
        config = self.knowledge_source_config

        # Create tools list
        tools = [
            # ReadFileTool(),
            # WriteFileTool(), 
            # ListDirectoryTool()
        ]
        
        # Add ynotebook tools if available
        if self.ynotebook_tools:
            tools.append(self.ynotebook_tools.tools)

        curator_agent = Agent(
            name="knowledge_curator",
            role=f"Knowledge curator for {config['source_name']}",
            model=AwsBedrock(id=model_id),
            instructions=self.create_generic_agent_instructions(user_input, notebook_context, config),
            tools=tools,
            markdown=True,
            show_tool_calls=True
        )
        
        return curator_agent

    async def process_message(self, message: Message):
        """
        Process user message and create knowledge file.
        
        Args:
            message: Incoming user message
        """
        try:
            message_text = message.body
            
            # Analyze notebook context
            notebook_context = self.analyze_notebook_context(message_text)
            
            # Create and run curator agent
            curator = self.create_curator_agent(message_text, notebook_context)
            
            config = self.knowledge_source_config
            task_description = f"""
                    Create a comprehensive knowledge file from {config['source_name']} to help with: {message_text}

                    Current cell context: {notebook_context['current_cell'] or 'No specific code provided'}
                    Notebook has {notebook_context['cell_count']} cells total.

                    Focus on providing practical, immediately useful information that addresses the user's specific question.
                    """
            
            result = curator.run(task_description, stream=False)
            response = result.content
            
        except Exception as e:
            logger.error(f"Error in knowledge curator: {e}")
            config = self.knowledge_source_config
            response = f"""I encountered an error while curating knowledge: {str(e)}

                    Please try again with a more specific question about {config['domain']}.

                    Example questions:
                    - "Explain pandas groupby operations with examples"
                    - "How do I create matplotlib subplots?"
                    - "Show me numpy array broadcasting"
                    """

        async def response_iterator():
            yield response

        await self.stream_message(response_iterator())


# Concrete implementation for Python Data Science Handbook
class PythonDataScienceHandbookCurator(ContextCuratorPersona):
    """
    Knowledge curator specifically for the Python Data Science Handbook.
    """
    
    @property
    def knowledge_source_config(self):
        return {
            'base_url': 'https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks',
            'source_name': 'Python Data Science Handbook',
            'domain': 'data science with Python (pandas, numpy, matplotlib, scikit-learn)',
            'description': 'analyzes code and questions to create curated data science knowledge files'
        }


# Additional example implementations
class PythonDocumentationCurator(ContextCuratorPersona):
    """
    Knowledge curator for official Python documentation.
    """
    
    @property
    def knowledge_source_config(self):
        return {
            'base_url': 'https://docs.python.org/3/',
            'source_name': 'Python Documentation',
            'domain': 'Python programming language and standard library',
            'description': 'creates knowledge files from official Python documentation'
        }


class StackOverflowCurator(ContextCuratorPersona):
    """
    Knowledge curator for Stack Overflow content.
    """
    
    @property
    def knowledge_source_config(self):
        return {
            'base_url': 'https://api.stackexchange.com/2.3/',
            'source_name': 'Stack Overflow',
            'domain': 'programming questions and community solutions',
            'description': 'finds and curates programming solutions from Stack Overflow'
        }