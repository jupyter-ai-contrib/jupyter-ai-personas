
"""
Context Retrieval Specialist Persona - Simplified Version

Analyzes user prompts and jupyter notebook code to understand their current work and objectives,
then searches through the Python Data Science Handbook using RAG to find the most relevant 
documentation, examples, best practices, and technical resources.
"""

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.team.team import Team
from agno.tools.file import FileTools
import boto3
from langchain_core.messages import HumanMessage
from .file_reader_tool import NotebookReaderTool

# Import RAG functionality - simple import with fallback
try:
    from .rag_integration_tool import create_simple_rag_tools
    print("✅ RAG tools loaded successfully")
except ImportError:
    print("⚠️ RAG tools not available, using FileTools fallback")
    create_simple_rag_tools = None

session = boto3.Session()


class ContextRetrieverPersona(BasePersona):
    """
    Context Retrieval Specialist that analyzes prompts and notebook content
    to find relevant documentation and resources using RAG.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def defaults(self):
        return PersonaDefaults(
            name="ContextRetrieverPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Context retrieval specialist for data science projects. Analyzes prompts and notebooks to find relevant resources using RAG.",
            system_prompt="""I am a context retrieval specialist team that analyzes your data science work and finds relevant resources from the Python Data Science Handbook using RAG search. 

                My team consists of:
                - NotebookAnalyzer: Analyzes your current notebook content and context
                - KnowledgeSearcher: Uses RAG to find relevant handbook examples and documentation
                - MarkdownGenerator: Creates comprehensive reports with actionable recommendations

                I can help with:
                - Finding relevant code examples for your current analysis stage
                - Semantic search through the Python Data Science Handbook
                - Context-aware recommendations based on your notebook content
                - Best practices and patterns for data science workflows

                To use me:
                - Provide your prompt or objective
                - Include: notebook: /path/to/notebook.ipynb
                - I'll create a comprehensive markdown report with relevant handbook content""",
        )

    def get_knowledge_tools(self):
        """Get knowledge search tools - RAG if available, FileTools as fallback."""
        if create_simple_rag_tools:
            try:
                return [create_simple_rag_tools()]
            except:
                pass
        
        # Fallback to FileTools
        return [FileTools()]

    def initialize_context_retrieval_team(self, system_prompt: str):
        """Initialize the 3-agent context retrieval team."""
        model_id = self.config_manager.lm_provider_params["model_id"]
        # Initialize tools
        notebook_tools = [NotebookReaderTool()]
        knowledge_tools = self.get_knowledge_tools()
        
        # 1. NotebookAnalyzer Agent
        notebook_analyzer = Agent(
            name="NotebookAnalyzer",
            role="Notebook analysis specialist that extracts context for search",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Use extract_rag_context tool to read notebook content - do NOT generate new code",
                "Look for notebook path in user prompt (format: 'notebook: /path/to/file.ipynb')",
                "If no path provided, use: /Users/jujonahj/jupyter-ai-personas/jupyter_ai_personas/data_science_persona/test_context_retrieval.ipynb",
                "Extract notebook context including:",
                "- Libraries being used (pandas, numpy, sklearn, matplotlib, etc.)",
                "- Analysis stage: data_loading, eda, preprocessing, modeling, evaluation, visualization", 
                "- Data characteristics and problem domain",
                "- Current objectives and next steps",
                "Create structured context summary for the KnowledgeSearcher"
            ],
            tools=notebook_tools,
            markdown=True,
            show_tool_calls=True
        )
        
        # 2. KnowledgeSearcher Agent
        knowledge_searcher = Agent(
            name="KnowledgeSearcher",
            role="Repository search specialist that finds relevant handbook content",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Use available search tools to find relevant Python Data Science Handbook content",
                "Receive context from NotebookAnalyzer (libraries, stage, objectives)",
                "Generate multiple targeted searches based on the context:",
                "- Primary objective searches",
                "- Library-specific searches", 
                "- Analysis stage searches",
                "- Problem domain searches",
                "Find code examples, explanations, and best practices",
                "Focus on content matching the detected libraries and analysis stage"
            ],
            tools=knowledge_tools,
            markdown=True,
            show_tool_calls=True
        )
        
        # 3. MarkdownGenerator Agent
        markdown_generator = Agent(
            name="MarkdownGenerator", 
            role="Content synthesis specialist that creates markdown reports",
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                "Create comprehensive markdown reports using search results",
                "Structure with sections:",
                "- Executive Summary",
                "- Current Notebook Analysis", 
                "- Relevant Resources",
                "- Code Examples",
                "- Actionable Next Steps",
                "Include relevant code snippets with proper formatting",
                "Provide specific next steps based on current analysis stage",
                "Focus on actionable insights for immediate application"
            ],
            tools=[FileTools()],
            markdown=True,
            show_tool_calls=True
        )
        
        # Create team
        context_team = Team(
            name="context-retrieval-team",
            mode="coordinate",
            members=[notebook_analyzer, knowledge_searcher, markdown_generator],
            model=AwsBedrock(id=model_id, session=session),
            instructions=[
                f"Context Retrieval Session: {system_prompt}",
                "WORKFLOW:",
                "1. NotebookAnalyzer: Extract context from user prompt + notebook",
                "2. KnowledgeSearcher: Search handbook for relevant content",
                "3. MarkdownGenerator: Create comprehensive markdown report",
                "Focus on providing actionable recommendations"
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True,
            show_tool_calls=True
        )
        
        return context_team

    async def process_message(self, message: Message):
        """Process messages using the context retrieval team."""
        print(f"🚀 CONTEXT RETRIEVAL REQUEST: {message.body}")
        message_text = message.body

        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]
        
        # Get chat history
        history = YChatHistory(ychat=self.ychat, k=2)
        messages = await history.aget_messages()

        history_text = ""
        if messages:
            history_text = "\nPrevious conversation:\n"
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"

        # Create system prompt
        system_prompt = f"""
Context Retrieval Session:
Model: {model_id}
Provider: {provider_name}
User Request: {message_text}
{history_text}

Goal: Analyze notebook context and find relevant Python Data Science Handbook content.
"""

        # Initialize and run team
        context_team = self.initialize_context_retrieval_team(system_prompt)
        
        try:
            response = context_team.run(
                message_text,
                stream=False,
                stream_intermediate_steps=True,
                show_full_reasoning=True,
            )
            
            response_content = response.content
            
        except Exception as e:
            print(f"❌ Team execution error: {e}")
            response_content = f"Error in context retrieval: {str(e)}\n\nPlease try again or check the logs for more details."

        async def response_iterator():
            yield response_content
        
        await self.stream_message(response_iterator())