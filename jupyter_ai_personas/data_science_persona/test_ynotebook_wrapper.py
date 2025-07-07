#!/usr/bin/env python3
"""
Test YNotebook wrapper functionality.
"""

import sys
from unittest.mock import Mock
from ynotebook_wrapper import YNotebookToolsWrapper, create_read_notebook_cell_tool

def test_ynotebook_wrapper_without_notebook():
    """Test that YNotebook wrapper handles None gracefully."""
    
    wrapper = YNotebookToolsWrapper(None)
    
    # Test read operations
    result = wrapper.read_current_cell()
    assert "No notebook instance available" in result
    
    result = wrapper.read_cell(0)
    assert "No notebook instance available" in result
    
    # Test write operations
    result = wrapper.write_cell(0, "print('hello')")
    assert "No notebook instance available" in result
    
    # Test notebook info
    info = wrapper.get_notebook_info()
    assert info['cell_count'] == 0
    assert info['has_cells'] == False
    
    print("✓ YNotebook wrapper without notebook test passed")

def test_agno_tool_integration():
    """Test that Agno tools are properly integrated."""
    
    wrapper = YNotebookToolsWrapper(None)
    
    # Test tool creation
    read_tool = create_read_notebook_cell_tool(wrapper)
    assert read_tool.name == "read_notebook_cell"
    assert "read" in read_tool.description.lower()
    
    # Test tool execution - use entrypoint which is the actual function
    result = read_tool.entrypoint(0)
    assert "No notebook instance available" in result
    
    print("✓ Agno tool integration test passed")

def test_wrapper_methods():
    """Test wrapper methods handle None gracefully."""
    
    wrapper = YNotebookToolsWrapper(None)
    
    # Test search
    results = wrapper.search_cells("pandas")
    assert results == []
    
    # Test add cell
    result = wrapper.add_cell(0, "code")
    assert "No notebook instance available" in result
    
    # Test delete cell
    result = wrapper.delete_cell(0)
    assert "No notebook instance available" in result
    
    print("✓ Wrapper methods test passed")

if __name__ == "__main__":
    try:
        test_ynotebook_wrapper_without_notebook()
        test_agno_tool_integration()
        test_wrapper_methods()
        print("\n🎉 All YNotebook wrapper tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)