"""
ynotebook_wrapper.py

Agno wrapper for YNotebook tools to enable cell manipulation in Jupyter AI personas.
This wrapper provides direct YNotebook manipulation without dependency on jupyter_ai_tools.
"""

import asyncio
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from jupyter_ydoc import YNotebook
from pathlib import Path


class NotebookContext:
    """Holds context information about the current notebook"""
    
    def __init__(self):
        self.notebook_path: Optional[str] = None
        self.kernel_id: Optional[str] = None
        self.last_activity: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
    
    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert context to dictionary"""
        return {
            'notebook_path': self.notebook_path,
            'kernel_id': self.kernel_id,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'metadata': self.metadata
        }


class YNotebookTools:
    """
    Direct YNotebook manipulation tools for Jupyter AI personas.
    
    This class works directly with YNotebook objects instead of file paths,
    providing a clean interface for notebook cell operations.
    """
    
    def __init__(self, ynotebook: YNotebook):
        """
        Initialize with a YNotebook instance.
        
        Args:
            ynotebook: The YNotebook instance to operate on
        """
        self.ynotebook = ynotebook
        self.context = NotebookContext()
        self._initialize_context()

    def _initialize_context(self):
        """Initialize notebook context from available sources."""
        # Try to get notebook path from YNotebook metadata
        try:
            if hasattr(self.ynotebook, 'path'):
                self.context.notebook_path = self.ynotebook.path
                self.context.metadata['source'] = 'ynotebook'
            elif hasattr(self.ynotebook, 'metadata'):
                # Try to extract from metadata
                metadata = self.ynotebook.metadata
                if isinstance(metadata, dict) and 'path' in metadata:
                    self.context.notebook_path = metadata['path']
                    self.context.metadata['source'] = 'metadata'
        except:
            pass
        
        self.context.update_activity()

    def get_notebook_data(self) -> dict:
        """
        Get the notebook data as a dictionary.
        
        Returns:
            dict: The notebook data including cells
        """
        try:
            # Access the notebook's cell data
            cells = []
            
            # YNotebook has a ycells attribute that contains the cells
            if hasattr(self.ynotebook, 'ycells'):
                for i, ycell in enumerate(self.ynotebook.ycells):
                    cell_dict = {
                        'cell_type': ycell.get('cell_type', 'code'),
                        'source': ycell.get('source', ''),
                        'metadata': ycell.get('metadata', {}),
                        'id': ycell.get('id', str(i))
                    }
                    cells.append(cell_dict)
            
            return {
                'cells': cells,
                'metadata': getattr(self.ynotebook, 'metadata', {}),
                'nbformat': 4,
                'nbformat_minor': 5
            }
        except Exception as e:
            return {'cells': [], 'error': str(e)}

    def read_cell_content(self, index: int) -> str:
        """
        Read the content of a specific cell.
        
        Args:
            index: The cell index to read
            
        Returns:
            str: Cell content as JSON string or error message
        """
        try:
            self.context.update_activity()
            
            if hasattr(self.ynotebook, 'ycells'):
                if 0 <= index < len(self.ynotebook.ycells):
                    cell = self.ynotebook.ycells[index]
                    cell_data = {
                        'cell_type': cell.get('cell_type', 'code'),
                        'source': cell.get('source', ''),
                        'metadata': cell.get('metadata', {}),
                        'id': cell.get('id', str(index))
                    }
                    return json.dumps(cell_data)
                else:
                    return f"❌ Cell index {index} out of range (0-{len(self.ynotebook.ycells)-1})"
            
            return "❌ No cells found in notebook"
        except Exception as e:
            return f"❌ Error reading cell {index}: {str(e)}"

    def get_cell_source(self, index: int) -> str:
        """
        Get just the source code from a cell.
        
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
        try:
            self.context.update_activity()
            
            if hasattr(self.ynotebook, 'ycells'):
                if 0 <= index < len(self.ynotebook.ycells):
                    # Update the cell content
                    self.ynotebook.ycells[index]['source'] = content
                    return f"✅ Successfully updated cell {index}"
                else:
                    return f"❌ Cell index {index} out of range"
            
            return "❌ No cells found in notebook"
        except Exception as e:
            return f"❌ Error writing to cell {index}: {str(e)}"

    def add_new_cell(self, index: int, cell_type: str = "code") -> str:
        """
        Add a new cell at the specified index.
        
        Args:
            index: Where to insert the new cell
            cell_type: Type of cell ("code" or "markdown")
            
        Returns:
            str: Success or error message
        """
        try:
            self.context.update_activity()
            
            if hasattr(self.ynotebook, 'ycells'):
                new_cell = {
                    'cell_type': cell_type,
                    'source': '',
                    'metadata': {},
                    'id': f'cell_{index}_{datetime.now().timestamp()}'
                }
                
                # Insert at the specified index
                self.ynotebook.ycells.insert(index, new_cell)
                return f"✅ Successfully added {cell_type} cell at index {index}"
            
            return "❌ Unable to add cell - notebook structure not found"
        except Exception as e:
            return f"❌ Error adding cell: {str(e)}"

    def remove_cell(self, index: int) -> str:
        """
        Delete a cell at the specified index.
        
        Args:
            index: The cell index to delete
            
        Returns:
            str: The deleted cell content or error message
        """
        try:
            self.context.update_activity()
            
            if hasattr(self.ynotebook, 'ycells'):
                if 0 <= index < len(self.ynotebook.ycells):
                    deleted_cell = self.ynotebook.ycells.pop(index)
                    return json.dumps(deleted_cell)
                else:
                    return f"❌ Cell index {index} out of range"
            
            return "❌ No cells found in notebook"
        except Exception as e:
            return f"❌ Error deleting cell {index}: {str(e)}"

    def get_notebook_content(self) -> str:
        """
        Get the full notebook content.
        
        Returns:
            str: JSON-formatted notebook content or error message
        """
        try:
            notebook_data = self.get_notebook_data()
            return json.dumps(notebook_data)
        except Exception as e:
            return f"❌ Error getting notebook content: {str(e)}"

    def get_cell_count(self) -> int:
        """
        Get the total number of cells in the notebook.
        
        Returns:
            int: Number of cells or -1 on error
        """
        try:
            if hasattr(self.ynotebook, 'ycells'):
                return len(self.ynotebook.ycells)
            return 0
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

    def get_active_notebook_path(self) -> Optional[str]:
        """Get the path of the currently active notebook."""
        return self.context.notebook_path

    def set_active_notebook_path(self, path: str):
        """Manually set the active notebook path."""
        self.context.notebook_path = path
        self.context.metadata['source'] = 'manual'
        self.context.update_activity()

    def get_notebook_context(self) -> dict:
        """Get full context information about the notebook."""
        return self.context.to_dict()


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
    
    def get_active_notebook_info(self) -> dict:
        """Get information about the currently active notebook."""
        context = self.tools.get_notebook_context()
        info = self.get_notebook_info()
        
        return {
            'active': True if context['notebook_path'] else False,
            'path': context['notebook_path'],
            'kernel_id': context['kernel_id'],
            'last_activity': context['last_activity'],
            'cell_count': info['cell_count'],
            'detection_source': context['metadata'].get('source', 'unknown'),
            'metadata': context['metadata']
        }
    
    def set_notebook_context(self, path: Optional[str] = None, kernel_id: Optional[str] = None):
        """Manually set notebook context information."""
        if path:
            self.tools.set_active_notebook_path(path)
        if kernel_id:
            self.tools.context.kernel_id = kernel_id
    
    def read_cell(self, index: int) -> str:
        """Read a specific cell by index."""
        return self.tools.get_cell_source(index)
    
    def write_cell(self, index: int, content: str, stream: bool = False) -> str:
        """Write content to a specific cell."""
        return self.tools.write_cell_content(index, content, stream)
    
    def add_cell(self, index: int, cell_type: str = "code") -> str:
        """Add a new cell."""
        return self.tools.add_new_cell(index, cell_type)
    
    def delete_cell(self, index: int) -> str:
        """Delete a cell."""
        return self.tools.remove_cell(index)
    
    def search_cells(self, text: str) -> List[int]:
        """Find cells containing specific text."""
        return self.tools.find_cells_with_content(text)
    
    def get_notebook_info(self) -> dict:
        """Get basic information about the notebook."""
        cell_count = self.tools.get_cell_count()
        return {
            'cell_count': cell_count,
            'max_index': cell_count - 1 if cell_count > 0 else -1,
            'has_cells': cell_count > 0
        }
    
    def get_notebook_summary(self) -> str:
        """Get a formatted summary of the notebook including context."""
        active_info = self.get_active_notebook_info()
        
        summary = f"📓 Notebook Status:\n"
        
        if active_info['active']:
            summary += f"✅ Active notebook: {active_info['path'] or 'Unknown path'}\n"
            summary += f"   Detection method: {active_info['detection_source']}\n"
        else:
            summary += "❓ No active notebook detected\n"
        
        if active_info['kernel_id']:
            summary += f"🔧 Kernel ID: {active_info['kernel_id']}\n"
        
        summary += f"📊 Cells: {active_info['cell_count']} total\n"
        
        if active_info['last_activity']:
            summary += f"⏰ Last activity: {active_info['last_activity']}\n"
        
        return summary
    
    def get_tools(self) -> list:
        """Get list of tools for Agno agent integration."""
        # TODO: Implement proper Agno tools integration
        return []