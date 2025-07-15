"""
rag_integration_tool.py

Agno tool wrapper for the Python Data Science Handbook RAG system.
Provides clean integration with Agno agents and error handling.
"""

from agno.tools import Toolkit
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path

# Import our core RAG system
try:
    from .rag_core import PythonDSHandbookRAG, create_handbook_rag
    RAG_CORE_AVAILABLE = True
except ImportError:
    try:
        from rag_core import PythonDSHandbookRAG, create_handbook_rag
        RAG_CORE_AVAILABLE = True
    except ImportError:
        RAG_CORE_AVAILABLE = False
        logging.error("rag_core module not found!")

logger = logging.getLogger(__name__)


class RAGSearchTool(Toolkit):
    """Agno tool for searching Python Data Science Handbook using RAG."""
    
    def __init__(self, force_rebuild: bool = False, **kwargs):
        """
        Initialize RAG search tool.
        
        Args:
            force_rebuild: Whether to force rebuild the vector store
            **kwargs: Additional arguments for RAG system
        """
        super().__init__(name="rag_search")
        self.rag_system = None
        self.force_rebuild = force_rebuild
        self.initialization_error = None
        
        # Initialize RAG system
        self._initialize_rag_system()
        
        # Register tool methods
        self.register(self.search_repository)
        self.register(self.search_by_topic)
        self.register(self.search_code_examples)
        self.register(self.get_system_status)
        self.register(self.rebuild_vector_store)
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with error handling."""
        if not RAG_CORE_AVAILABLE:
            self.initialization_error = "RAG core module not available"
            logger.error(self.initialization_error)
            return
        
        try:
            logger.info("Initializing Python Data Science Handbook RAG system...")
            self.rag_system = create_handbook_rag(force_rebuild=self.force_rebuild)
            
            if self.rag_system:
                logger.info("✅ RAG system initialized successfully")
            else:
                self.initialization_error = "RAG system initialization returned None"
                logger.error(self.initialization_error)
                
        except Exception as e:
            self.initialization_error = f"RAG initialization failed: {str(e)}"
            logger.error(self.initialization_error)
    
    def search_repository(self, query: str, k: int = 5, include_scores: bool = False) -> str:
        """
        Search the Python Data Science Handbook repository.
        
        Args:
            query: Search query (e.g., "pandas groupby operations")
            k: Number of results to return (default: 5)
            include_scores: Whether to include similarity scores
            
        Returns:
            JSON string with search results
        """
        if not self.rag_system:
            return json.dumps({
                "error": f"RAG system not available: {self.initialization_error}",
                "query": query,
                "results": []
            })
        
        try:
            if include_scores:
                raw_results = self.rag_system.search_with_scores(query, k=k)
                results = [
                    {
                        "content": result[0]["content"],
                        "source": result[0]["source"],
                        "notebook_name": result[0]["notebook_name"],
                        "cell_type": result[0]["cell_type"],
                        "similarity_score": float(result[1]),
                        "metadata": result[0]["metadata"]
                    }
                    for result in raw_results
                ]
            else:
                raw_results = self.rag_system.search(query, k=k)
                results = [
                    {
                        "content": result["content"],
                        "source": result["source"],
                        "notebook_name": result["notebook_name"],
                        "cell_type": result["cell_type"],
                        "metadata": result["metadata"]
                    }
                    for result in raw_results
                ]
            
            response = {
                "query": query,
                "total_results": len(results),
                "results": results,
                "search_successful": True
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            error_response = {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": [],
                "search_successful": False
            }
            return json.dumps(error_response)
    
    def search_by_topic(self, topic: str, notebook_context: str = None, k: int = 7) -> str:
        """
        Search for content related to a specific data science topic.
        
        Args:
            topic: Topic to search for (e.g., "data cleaning", "visualization", "machine learning")
            notebook_context: Optional context from current notebook analysis
            k: Number of results to return
            
        Returns:
            JSON string with topic-specific results
        """
        if not self.rag_system:
            return json.dumps({
                "error": f"RAG system not available: {self.initialization_error}",
                "topic": topic,
                "results": []
            })
        
        try:
            # Create enhanced search queries for the topic
            search_queries = [
                topic,
                f"{topic} python examples",
                f"{topic} tutorial step by step",
                f"how to {topic}"
            ]
            
            all_results = []
            seen_content = set()
            
            for query in search_queries:
                results = self.rag_system.search(query, k=max(2, k//len(search_queries)))
                
                for result in results:
                    # Avoid duplicate content
                    content_hash = hash(result["content"][:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_results.append(result)
            
            # Sort by relevance if we have scores, otherwise keep order
            final_results = all_results[:k]
            
            response = {
                "topic": topic,
                "search_queries_used": search_queries,
                "total_results": len(final_results),
                "results": final_results,
                "notebook_context_applied": notebook_context is not None
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            error_response = {
                "error": f"Topic search failed: {str(e)}",
                "topic": topic,
                "results": []
            }
            return json.dumps(error_response)
    
    def search_code_examples(self, task_description: str, libraries: List[str] = None, k: int = 5) -> str:
        """
        Search specifically for code examples related to a task.
        
        Args:
            task_description: What the user wants to accomplish
            libraries: List of libraries they're using (e.g., ["pandas", "matplotlib"])
            k: Number of code examples to return
            
        Returns:
            JSON string with code examples
        """
        if not self.rag_system:
            return json.dumps({
                "error": f"RAG system not available: {self.initialization_error}",
                "task": task_description,
                "results": []
            })
        
        try:
            # Build search query with libraries if provided
            if libraries:
                library_str = " ".join(libraries)
                search_query = f"{task_description} {library_str} code example"
            else:
                search_query = f"{task_description} python code example"
            
            # Search for results
            results = self.rag_system.search(search_query, k=k*2)  # Get more to filter
            
            # Filter for code cells and relevant content
            code_results = []
            for result in results:
                # Prioritize code cells
                if result["cell_type"] == "code" or "```" in result["content"] or "import " in result["content"]:
                    code_results.append(result)
                elif len(code_results) < k:  # Include markdown if we need more examples
                    code_results.append(result)
            
            # Limit to requested number
            final_results = code_results[:k]
            
            response = {
                "task_description": task_description,
                "libraries_requested": libraries or [],
                "search_query": search_query,
                "total_results": len(final_results),
                "results": final_results,
                "code_examples_found": len([r for r in final_results if r["cell_type"] == "code"])
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            error_response = {
                "error": f"Code search failed: {str(e)}",
                "task": task_description,
                "results": []
            }
            return json.dumps(error_response)
    
    def get_system_status(self) -> str:
        """Get detailed status of the RAG system for debugging."""
        if not self.rag_system:
            status = {
                "rag_system_available": False,
                "initialization_error": self.initialization_error,
                "core_module_available": RAG_CORE_AVAILABLE
            }
        else:
            status = self.rag_system.get_stats()
            status["rag_system_available"] = True
            status["initialization_error"] = None
        
        return json.dumps(status, indent=2)
    
    def rebuild_vector_store(self) -> str:
        """Force rebuild the vector store (useful if repository was updated)."""
        try:
            logger.info("Force rebuilding vector store...")
            
            if not RAG_CORE_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "RAG core module not available"
                })
            
            # Reinitialize with force rebuild
            self.rag_system = create_handbook_rag(force_rebuild=True)
            
            if self.rag_system:
                return json.dumps({
                    "success": True,
                    "message": "Vector store rebuilt successfully",
                    "stats": self.rag_system.get_stats()
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "Failed to rebuild RAG system"
                })
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Rebuild failed: {str(e)}"
            })


# Factory function for easy initialization
def create_simple_rag_tools(force_rebuild: bool = False) -> RAGSearchTool:
    """
    Create RAG tools for the Context Retrieval Persona.
    
    Args:
        force_rebuild: Whether to force rebuild the vector store
        
    Returns:
        RAGSearchTool instance ready for use with Agno agents
    """
    return RAGSearchTool(force_rebuild=force_rebuild)


# Quick test function
def test_rag_integration():
    """Test the RAG integration tool."""
    print("🧪 Testing RAG integration tool...")
    
    try:
        rag_tool = create_simple_rag_tools()
        
        # Test basic search
        result = rag_tool.search_repository("pandas dataframe", k=2)
        result_data = json.loads(result)
        
        if result_data.get("search_successful"):
            print("✅ RAG integration test successful!")
            print(f"Found {result_data['total_results']} results")
            return True
        else:
            print(f"❌ RAG integration test failed: {result_data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ RAG integration test failed with exception: {e}")
        return False


if __name__ == "__main__":
    test_rag_integration()