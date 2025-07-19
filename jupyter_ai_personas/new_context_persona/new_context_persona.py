"""
New Context Retrieval Persona using PocketFlow Architecture

Simple implementation that uses existing RAG tools orchestrated by PocketFlow.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from langchain_core.messages import HumanMessage
from agno.tools.file import FileTools

# Import existing RAG tools from original persona
try:
    from ..context_retrieval_persona.rag_integration_tool import create_simple_rag_tools
    from ..context_retrieval_persona.file_reader_tool import NotebookReaderTool
    print("✅ Existing RAG and notebook tools loaded successfully")
    RAG_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import existing tools: {e}")
    RAG_TOOLS_AVAILABLE = False

# Import our PocketFlow architecture
from .context_flow import create_context_retrieval_flow

logger = logging.getLogger(__name__)


class NewContextPersona(BasePersona):
    """
    New Context Retrieval Persona using PocketFlow Architecture
    
    Combines the existing RAG tools with PocketFlow orchestration
    and adds conversational capabilities like the data science persona.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize tools using existing infrastructure
        self.notebook_tools = [NotebookReaderTool()] if RAG_TOOLS_AVAILABLE else []
        
        # Initialize RAG tools with error handling
        self.rag_tools = []
        if RAG_TOOLS_AVAILABLE:
            try:
                rag_tool = create_simple_rag_tools()
                self.rag_tools = [rag_tool]
                logger.info(f"✅ RAG tool initialized: {type(rag_tool).__name__}")
            except Exception as e:
                logger.error(f"❌ RAG tool initialization failed: {e}")
                self.rag_tools = []
        
        self.file_tools = [FileTools()]
        
        # Initialize PocketFlow
        self.context_flow = create_context_retrieval_flow(
            notebook_tools=self.notebook_tools,
            rag_tools=self.rag_tools,
            file_tools=self.file_tools
        )
        
        logger.info("✅ NewContextPersona initialized with PocketFlow architecture")
    
    @property
    def defaults(self):
        return PersonaDefaults(
            name="NewContextPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Context retrieval specialist using PocketFlow architecture. Analyzes notebooks and provides RAG-based recommendations.",
            system_prompt="""I am a context retrieval specialist powered by PocketFlow architecture that combines existing RAG tools with intelligent orchestration.

My capabilities:
🔍 **Notebook Analysis** - I analyze your Jupyter notebook content, libraries, and analysis stage
📚 **RAG-based Search** - I search the Python Data Science Handbook using existing, proven tools  
💡 **Context-Aware Recommendations** - I provide targeted suggestions based on your work
📝 **Comprehensive Reports** - I generate detailed markdown reports with actionable insights

I use PocketFlow to orchestrate the same reliable components from the original context retrieval persona:
- NotebookAnalyzer: Extracts context from your notebooks
- KnowledgeSearcher: Uses proven RAG tools to find relevant content
- MarkdownGenerator: Creates comprehensive reports

I'm also conversational! I can:
- Respond to greetings and casual questions
- Understand your intent and respond appropriately  
- Provide simple answers for quick questions
- Run full analysis for complex requests

To use me:
- Just ask questions about your data science work
- Include `notebook: /path/to/file.ipynb` for notebook-specific analysis
- I work great with pandas, numpy, matplotlib, seaborn, sklearn questions

What would you like help with today?""",
        )
    
    async def process_message(self, message: Message):
        """Process messages with conversational intelligence and PocketFlow orchestration."""
        try:
            logger.info(f"🧠 NEW CONTEXT PERSONA: {message.body}")
            message_text = message.body.strip()
            
            # Get chat history for context
            history = YChatHistory(ychat=self.ychat, k=3)
            messages = await history.aget_messages()
            
            # Agent Brain: Analyze intent and decide response strategy
            response_strategy = self._analyze_message_intent(message_text, messages)
            
            # Route to appropriate handler
            if response_strategy["type"] == "greeting":
                response_content = self._handle_greeting(message_text, response_strategy)
            elif response_strategy["type"] == "simple_question":
                response_content = self._handle_simple_question(message_text, response_strategy)
            elif response_strategy["type"] == "context_analysis":
                response_content = self._handle_context_analysis(message_text, response_strategy)
            elif response_strategy["type"] == "status_check":
                response_content = self._handle_status_check(message_text, response_strategy)
            else:
                # Default to context analysis for comprehensive requests
                response_content = self._handle_context_analysis(message_text, response_strategy)
            
            # Stream response
            async def response_iterator():
                yield response_content
            
            await self.stream_message(response_iterator())
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            error_response = self._create_error_response(str(e))
            
            async def error_iterator():
                yield error_response
            
            await self.stream_message(error_iterator())
    
    def _analyze_message_intent(self, message_text: str, chat_history: list) -> Dict[str, Any]:
        """Simple intent analysis using heuristics."""
        message_lower = message_text.lower()
        
        # Greeting detection
        if any(word in message_lower for word in ["hello", "hi", "hey"]) and len(message_text.split()) <= 3:
            return {"type": "greeting", "context": "initial_greeting" if not chat_history else "continued_greeting"}
        
        # Status check detection
        if any(word in message_lower for word in ["status", "setup", "ready", "working"]):
            return {"type": "status_check"}
        
        # Context analysis detection (comprehensive requests)
        if any(indicator in message_text for indicator in [".ipynb", "analyze", "notebook:"]) or len(message_text) > 100:
            return {
                "type": "context_analysis",
                "notebook_path": self._extract_notebook_path(message_text),
                "analysis_depth": "comprehensive"
            }
        
        # Simple question detection
        if any(phrase in message_lower for phrase in ["what is", "how to", "explain", "show me"]) and len(message_text) < 100:
            return {"type": "simple_question", "requires_rag": True}
        
        # Default to context analysis for unclear requests
        return {"type": "context_analysis", "notebook_path": self._extract_notebook_path(message_text)}
    
    def _handle_greeting(self, message_text: str, strategy: Dict[str, Any]) -> str:
        """Handle greeting messages conversationally."""
        if strategy.get("context") == "initial_greeting":
            return """Hello! 👋 I'm your **Context Retrieval Specialist** using PocketFlow architecture.

I can help you with:
🔍 **Analyzing Jupyter notebooks** - I'll examine your code, libraries, and analysis stage
📚 **Finding relevant resources** - I search the Python Data Science Handbook using proven RAG tools
💡 **Providing recommendations** - Context-aware suggestions based on your current work
📝 **Creating detailed reports** - Comprehensive analysis with actionable next steps

**How to use me:**
- Ask questions about your data science work
- Include `notebook: /path/to/file.ipynb` for notebook-specific analysis
- I work great with pandas, numpy, sklearn, matplotlib, seaborn questions

What would you like help with today?"""
        else:
            return """Hi again! 👋 

I'm here and ready to help with your data science questions. What's on your mind?

💡 **Tip**: For the most helpful analysis, you can:
- Ask about specific libraries or techniques
- Share your notebook path for personalized recommendations
- Describe what you're trying to accomplish"""
    
    def _handle_status_check(self, message_text: str, strategy: Dict[str, Any]) -> str:
        """Handle status check requests."""
        status_report = "# System Status Check\n\n"
        
        # Check component availability
        components = {
            "PocketFlow Architecture": True,
            "RAG Tools": RAG_TOOLS_AVAILABLE and bool(self.rag_tools),
            "Notebook Reader": RAG_TOOLS_AVAILABLE and bool(self.notebook_tools),
            "File Tools": bool(self.file_tools)
        }
        
        all_good = all(components.values())
        if all_good:
            status_report += "✅ **All systems operational!**\n\n"
        else:
            status_report += "⚠️ **Some issues detected**\n\n"
        
        status_report += "## Component Status\n"
        for component, is_ok in components.items():
            indicator = "✅" if is_ok else "❌"
            status_report += f"- {component}: {indicator}\n"
        
        if not components["RAG Tools"]:
            status_report += "\n## Setup Required\n"
            status_report += "🔧 RAG tools need to be initialized. This will:\n"
            status_report += "- Set up the Python Data Science Handbook search\n"
            status_report += "- Enable full context retrieval capabilities\n\n"
            status_report += "Just ask me any question and I'll help set it up!"
        
        return status_report
    
    def _handle_simple_question(self, message_text: str, strategy: Dict[str, Any]) -> str:
        """Handle simple questions with light search."""
        try:
            if self.rag_tools and hasattr(self.rag_tools[0], 'search_repository'):
                # Quick search using existing tools
                result = self.rag_tools[0].search_repository(message_text, k=2)
                
                # Try to parse result
                import json
                try:
                    result_data = json.loads(result) if isinstance(result, str) else result
                    if result_data.get("search_successful") and result_data.get("results"):
                        docs = result_data["results"][:2]
                        
                        response = f"## {message_text}\n\n"
                        response += "Here's what I found in the Python Data Science Handbook:\n\n"
                        
                        for i, doc in enumerate(docs, 1):
                            content = doc.get("content", "")[:300] + "..." if doc.get("content") else "No content available"
                            notebook = doc.get("notebook_name", "Unknown")
                            response += f"**{i}. From {notebook}:**\n{content}\n\n"
                        
                        response += "💡 **Need more detailed help?** Ask for a full analysis or share your notebook path!"
                        return response
                except:
                    pass
            
            # Fallback for simple questions
            return f"""I'd like to help with: "{message_text}"

🔧 **Quick note**: For the best answers, I can run a full search through the Python Data Science Handbook.

**What I can do:**
- Find specific examples and tutorials
- Provide context-aware recommendations
- Analyze your notebooks for personalized advice

**To get detailed help:**
1. Ask for a full analysis (I'll search comprehensively)
2. Include your notebook path for personalized results
3. Be specific about what you're trying to accomplish

Would you like me to run a comprehensive search for your question?"""
            
        except Exception as e:
            logger.error(f"Simple question handling failed: {e}")
            return self._create_simple_fallback(message_text)
    
    def _handle_context_analysis(self, message_text: str, strategy: Dict[str, Any]) -> str:
        """Handle comprehensive context analysis using PocketFlow."""
        try:
            # Extract notebook path
            notebook_path = strategy.get("notebook_path") or self._extract_notebook_path(message_text)
            
            # Prepare shared data for PocketFlow
            shared_data = {
                "user_query": message_text,
                "notebook_path": notebook_path
            }
            
            # Run PocketFlow orchestration
            logger.info(f"🔄 Running PocketFlow context retrieval")
            final_result = self.context_flow.run(shared_data)
            
            # Extract final report from shared data
            final_report = shared_data.get("final_report", "")
            
            if final_report:
                # Add flow summary
                report_saved = shared_data.get("report_saved", False)
                
                summary = f"""🔄 **PocketFlow Analysis Complete**
- Flow execution: {'Success' if final_result == 'default' else 'Completed with issues'}
- Report generated: {'Yes' if report_saved else 'No'}

---

{final_report}"""
                return summary
            else:
                # Fallback formatting
                return self._format_flow_results(shared_data)
                
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return self._create_error_response(str(e))
    
    def _format_flow_results(self, result: Dict[str, Any]) -> str:
        """Format flow results when no final report is available."""
        user_query = result.get("user_query", "Unknown")
        notebook_analysis = result.get("notebook_analysis", {})
        search_results = result.get("search_results", [])
        
        response = f"""# PocketFlow Context Analysis

## Query: {user_query}

## Notebook Analysis
- **Path**: {notebook_analysis.get('notebook_path', 'Not specified')}
- **Libraries**: {', '.join(notebook_analysis.get('libraries', []))}
- **Stage**: {notebook_analysis.get('analysis_stage', 'Unknown')}

## Search Results
Found {len(search_results)} relevant searches through the handbook.

## Flow Execution Summary
"""
        
        flow_results = result.get("flow_results", [])
        for flow_result in flow_results:
            node_name = flow_result.get("node", "Unknown")
            success = flow_result.get("success", False)
            status = "✅" if success else "❌"
            response += f"- {node_name}: {status}\n"
        
        response += "\n## Recommendations\n"
        response += "Based on the analysis, consider:\n"
        response += "1. Reviewing relevant examples from the handbook\n"
        response += "2. Optimizing your current approach\n"
        response += "3. Following data science best practices\n"
        
        return response
    
    def _create_simple_fallback(self, message_text: str) -> str:
        """Create a simple fallback response."""
        return f"""I'd like to help with: "{message_text}"

**What I can do:**
- Analyze your notebooks using PocketFlow architecture
- Search the Python Data Science Handbook for relevant examples  
- Provide context-aware recommendations

**To get started:**
1. Ask any question (I'll use my full capabilities)
2. Include your notebook path for personalized analysis
3. Be specific about what you're trying to accomplish

What would you like to explore?"""
    
    def _create_error_response(self, error_msg: str) -> str:
        """Create a user-friendly error response."""
        return f"""🚨 **Oops! Something went wrong**

I encountered an issue: `{error_msg}`

**Let's try this:**
1. 🔄 **Rephrase your question** - Sometimes simpler is better
2. 📝 **Check notebook path** - If you provided one, make sure it's correct
3. ⚡ **Try a basic question** - Like "what is pandas?" to test the system
4. 🛠️ **System check** - Ask about "status" to see what's working

I'm here to help, so let's figure this out together! What would you like to try?"""
    
    def _extract_notebook_path(self, message_text: str) -> Optional[str]:
        """Extract notebook path from message text."""
        # Look for "notebook: path" pattern
        if "notebook:" in message_text.lower():
            parts = message_text.split("notebook:")
            if len(parts) > 1:
                path_part = parts[1].strip().split()[0]
                return path_part
        
        # Look for .ipynb file paths
        if ".ipynb" in message_text:
            words = message_text.split()
            for word in words:
                if word.endswith('.ipynb'):
                    return word
        
        return None