"""
Context Retrieval Flow Configuration

Combines the context nodes into a PocketFlow workflow.
"""

from .pocketflow import Flow
from .context_nodes import NotebookAnalysisNode, KnowledgeSearchNode, ReportGenerationNode


def create_context_retrieval_flow(notebook_tools, rag_tools, file_tools) -> Flow:
    """Create the main context retrieval flow using PocketFlow architecture."""
    
    # Create nodes
    notebook_node = NotebookAnalysisNode(notebook_tools)
    search_node = KnowledgeSearchNode(rag_tools)
    report_node = ReportGenerationNode(file_tools)
    
    # Chain nodes together
    notebook_node >> search_node >> report_node
    
    # Create and return flow
    flow = Flow(start=notebook_node)
    return flow