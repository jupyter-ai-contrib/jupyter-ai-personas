"""
ynotebook_wrapper.py

Agno wrapper for YNotebook tools to enable cell manipulation in Jupyter AI personas.
This wrapper converts async ynotebook functions into sync methods that work with Agno agents.
"""

import asyncio
import json
from typing import Optional, List
from jupyter_ydoc import YNotebook
#from jupyter_ai_tools.tools.toolkits.file_system import toolkit

# Import the ynotebook functions (adjust path as needed)
from jupyter_ai_tools.ynotebook_tools import (
    read_cell,
    write_to_cell,
    add_cell,
    delete_cell,
    read_notebook,
    get_max_cell_index
)


class YNotebookTools:
    """
    Agno wrapper for YNotebook tools to enable cell manipulation in Jupyter AI personas.
    
    This class wraps the async ynotebook functions and provides a sync interface
    that can be used with Agno agents.
    """
    
    def __init__(self, ynotebook: YNotebook):
        """
        Initialize with a YNotebook instance.
        
        Args:
            ynotebook: The YNotebook instance to operate on
        """
        self.ynotebook = ynotebook

    def _run_async(self, coro):
        """
        Helper method to run async functions in sync context.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
            else:
                return asyncio.run(coro)
        except Exception as e:
            return f"❌ Error running async operation: {str(e)}"

    def read_cell_content(self, index: int) -> str:
        """
        Read the content of a specific cell.
        
        Args:
            index: The cell index to read
            
        Returns:
            str: Cell content as JSON string or error message
        """
        return self._run_async(read_cell(self.ynotebook, index))

    def get_cell_source(self, index: int) -> str:
        """
        Get just the source code from a cell (not the full JSON).
        
        Args:
            index: The cell index to read
            
        Returns:
            str: The source code content of the cell
        """
        try:
            cell_json = self.read_cell_content(index)
            if cell_json.startswith("❌"):
                return cell_json
            
            cell_data = json.loads(cell_json)
            return cell_data.get('source', '')
        except Exception as e:
            return f"❌ Error extracting source from cell {index}: {str(e)}"

    def write_cell_content(self, index: int, content: str, stream: bool = True) -> str:
        """
        Write content to a specific cell.
        
        Args:
            index: The cell index to write to
            content: The content to write
            stream: Whether to simulate gradual updates
            
        Returns:
            str: Success or error message
        """
        return self._run_async(write_to_cell(self.ynotebook, index, content, stream))

    def add_new_cell(self, index: int, cell_type: str = "code") -> str:
        """
        Add a new cell at the specified index.
        
        Args:
            index: Where to insert the new cell
            cell_type: Type of cell ("code" or "markdown")
            
        Returns:
            str: Success or error message
        """
        print("called to add new cell")
        return self._run_async(add_cell(self.ynotebook, index, cell_type))

    def remove_cell(self, index: int) -> str:
        """
        Delete a cell at the specified index.
        
        Args:
            index: The cell index to delete
            
        Returns:
            str: The deleted cell content or error message
        """
        return self._run_async(delete_cell(self.ynotebook, index))

    def get_notebook_content(self) -> str:
        """
        Get the full notebook content.
        
        Returns:
            str: JSON-formatted notebook content or error message
        """
        return self._run_async(read_notebook(self.ynotebook))

    def get_cell_count(self) -> int:
        """
        Get the total number of cells in the notebook.
        
        Returns:
            int: Number of cells or -1 on error
        """
        try:
            max_index = self._run_async(get_max_cell_index(self.ynotebook))
            if isinstance(max_index, str) and max_index.startswith("❌"):
                return -1
            return max_index + 1  # Convert from max index to count
        except Exception:
            return -1

    def find_cells_with_content(self, search_text: str) -> List[int]:
        """
        Find all cells containing specific text.
        
        Args:
            search_text: Text to search for
            
        Returns:
            List[int]: List of cell indices that contain the search text
        """
        matching_cells = []
        cell_count = self.get_cell_count()
        
        if cell_count <= 0:
            return matching_cells
            
        for i in range(cell_count):
            source = self.get_cell_source(i)
            if not source.startswith("❌") and search_text.lower() in source.lower():
                matching_cells.append(i)
                
        return matching_cells

    def get_current_cell_index(self) -> Optional[int]:
        """
        Get the currently selected cell index.
        This would need to be implemented based on Jupyter AI's current cell tracking.
        
        Returns:
            Optional[int]: Current cell index or None if not available
        """
        # Placeholder - this would need integration with Jupyter AI's cell tracking
        # For now, return None to indicate this feature needs implementation
        return None


class YNotebookToolsWrapper:
    """
    Simplified wrapper for easy integration with Agno agents.
    Provides a clean, simple interface for common notebook operations.
    """
    
    def __init__(self, ynotebook: YNotebook):
        """
        Initialize with a YNotebook instance.
        
        Args:
            ynotebook: The YNotebook instance to operate on
        """
        self.tools = YNotebookTools(ynotebook)
    
    def read_current_cell(self) -> str:
        """
        Read the currently selected cell.
        
        Returns:
            str: Current cell source code or error message
        """
        current_index = self.tools.get_current_cell_index()
        if current_index is not None:
            return self.tools.get_cell_source(current_index)
        return "❌ No current cell selected"
    
    def read_cell(self, index: int) -> str:
        """
        Read a specific cell by index.
        
        Args:
            index: Cell index to read
            
        Returns:
            str: Cell source code
        """
        return self.tools.get_cell_source(index)
    
    def write_cell(self, index: int, content: str, stream: bool = False) -> str:
        """
        Write content to a specific cell.
        
        Args:
            index: Cell index to write to
            content: Content to write
            stream: Whether to animate the writing
            
        Returns:
            str: Success or error message
        """
        return self.tools.write_cell_content(index, content, stream)
    
    def add_cell(self, index: int, cell_type: str = "code") -> str:
        """
        Add a new cell.
        
        Args:
            index: Where to insert the cell
            cell_type: "code" or "markdown"
            
        Returns:
            str: Success or error message
        """
        return self.tools.add_new_cell(index, cell_type)
    
    def delete_cell(self, index: int) -> str:
        """
        Delete a cell.
        
        Args:
            index: Cell index to delete
            
        Returns:
            str: Deleted cell content or error message
        """
        return self.tools.remove_cell(index)
    
    def search_cells(self, text: str) -> List[int]:
        """
        Find cells containing specific text.
        
        Args:
            text: Text to search for
            
        Returns:
            List[int]: List of matching cell indices
        """
        return self.tools.find_cells_with_content(text)
    
    def get_notebook_info(self) -> dict:
        """
        Get basic information about the notebook.
        
        Returns:
            dict: Notebook information including cell count
        """
        cell_count = self.tools.get_cell_count()
        return {
            'cell_count': cell_count,
            'max_index': cell_count - 1 if cell_count > 0 else -1,
            'has_cells': cell_count > 0
        }