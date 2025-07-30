"""Enhanced file tools with structure awareness."""

import os
from agno.tools.file import FileTools as AgnoFileTools
from agno.agent import Agent

class EnhancedFileTools(AgnoFileTools):
    """Enhanced file tools with structure awareness."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repo_structure_tools = None
        self.repo_path = None
        
    def set_repo_path(self, repo_path):
        """Set the repository path for structure analysis."""
        self.repo_path = repo_path
        
    def set_structure_tools(self, structure_tools):
        """Set the repository structure tools instance."""
        self.repo_structure_tools = structure_tools
    
    def write_file(self, agent: Agent, path: str, content: str) -> str:
        """
        Write content to a file with structure validation.
        
        Args:
            agent: The agent instance
            path: Path to the file
            content: Content to write
            
        Returns:
            str: Result of the operation
        """
        # Validate file path if repo_structure_tools and repo_path are available
        if self.repo_structure_tools and self.repo_path:
            try:
                # Validate the path
                validation_result = self.repo_structure_tools.validate_file_path(agent, path, self.repo_path)
                
                # Check if path is directly in root
                if validation_result.startswith("WARNING:"):
                    # Extract suggested path from validation result
                    suggested_path = validation_result.split("Consider using: ")[1].strip()
                    print(f"WARNING: File would be created in repository root. Using suggested path: {suggested_path}")
                    path = suggested_path
            except Exception as e:
                print(f"Warning: Error validating file path: {str(e)}")
        
        # Ensure parent directories exist
        try:
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Created parent directories for: {path}")
        except Exception as e:
            print(f"Warning: Error creating parent directories: {str(e)}")
        
        # Call the original write_file method
        return super().write_file(agent, path, content)
    
    def append_file(self, agent: Agent, path: str, content: str) -> str:
        """
        Append content to a file with structure validation.
        
        Args:
            agent: The agent instance
            path: Path to the file
            content: Content to append
            
        Returns:
            str: Result of the operation
        """
        # Validate file path if repo_structure_tools and repo_path are available
        if self.repo_structure_tools and self.repo_path:
            try:
                # Validate the path
                validation_result = self.repo_structure_tools.validate_file_path(agent, path, self.repo_path)
                
                # Check if path is directly in root
                if validation_result.startswith("WARNING:"):
                    # Extract suggested path from validation result
                    suggested_path = validation_result.split("Consider using: ")[1].strip()
                    print(f"WARNING: File would be created in repository root. Using suggested path: {suggested_path}")
                    path = suggested_path
            except Exception as e:
                print(f"Warning: Error validating file path: {str(e)}")
        
        # Ensure parent directories exist
        try:
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Created parent directories for: {path}")
        except Exception as e:
            print(f"Warning: Error creating parent directories: {str(e)}")
        
        # Call the original append_file method
        return super().append_file(agent, path, content)
    
    def suggest_file_path(self, agent: Agent, file_name: str, component_type: str) -> str:
        """
        Suggest appropriate file path based on repository patterns.
        
        Args:
            agent: The agent instance
            file_name: Name of the file to create
            component_type: Type of component (model, view, controller, etc.)
            
        Returns:
            str: Suggested file path
        """
        if self.repo_structure_tools and self.repo_path:
            try:
                return self.repo_structure_tools.suggest_file_path(agent, file_name, component_type, self.repo_path)
            except Exception as e:
                print(f"Warning: Error suggesting file path: {str(e)}")
                
        # Default path if suggestion fails
        if self.base_path:
            return os.path.join(self.base_path, file_name)
        else:
            return file_name