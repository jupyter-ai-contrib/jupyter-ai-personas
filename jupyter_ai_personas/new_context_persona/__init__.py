"""
New Context Retrieval Persona Package

A simple PocketFlow-based context retrieval persona that uses existing RAG tools
orchestrated through a lightweight flow architecture.
"""

# Import the main persona
from .new_context_persona import NewContextPersona

# Import PocketFlow components
from .pocketflow import Flow, Node, BaseNode
from .context_flow import create_context_retrieval_flow
from .context_nodes import NotebookAnalysisNode, KnowledgeSearchNode, ReportGenerationNode

__all__ = [
    "NewContextPersona",
    "Flow",
    "Node", 
    "BaseNode",
    "create_context_retrieval_flow",
    "NotebookAnalysisNode", 
    "KnowledgeSearchNode",
    "ReportGenerationNode"
]