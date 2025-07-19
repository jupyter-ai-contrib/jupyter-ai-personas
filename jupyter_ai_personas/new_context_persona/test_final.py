"""
Final comprehensive test for the new context persona with proper PocketFlow architecture.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pocketflow_architecture():
    """Test that PocketFlow follows the original compact design."""
    try:
        from .pocketflow import Flow, Node, BaseNode, AsyncNode, BatchNode
        
        # Test basic structure
        flow = Flow()
        assert hasattr(flow, 'start_node')
        assert hasattr(flow, '_orch')
        
        node = Node()
        assert hasattr(node, 'prep')
        assert hasattr(node, 'exec') 
        assert hasattr(node, 'post')
        
        logger.info("✅ PocketFlow architecture test passed")
        return True
    except Exception as e:
        logger.error(f"❌ PocketFlow architecture test failed: {e}")
        return False

def test_context_nodes():
    """Test context-specific node implementations."""
    try:
        from .context_nodes import NotebookAnalysisNode, KnowledgeSearchNode, ReportGenerationNode
        from .pocketflow import Node
        
        # Test node creation
        notebook_node = NotebookAnalysisNode([])
        search_node = KnowledgeSearchNode([])
        report_node = ReportGenerationNode([])
        
        # Test inheritance
        assert isinstance(notebook_node, Node)
        assert isinstance(search_node, Node)
        assert isinstance(report_node, Node)
        
        logger.info("✅ Context nodes test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Context nodes test failed: {e}")
        return False

def test_flow_creation():
    """Test flow creation and chaining."""
    try:
        from .context_flow import create_context_retrieval_flow
        
        mock_tools = []
        flow = create_context_retrieval_flow(mock_tools, mock_tools, mock_tools)
        
        # Test flow structure
        assert flow.start_node is not None
        assert hasattr(flow.start_node, 'successors')
        assert len(flow.start_node.successors) > 0  # Should have next node
        
        logger.info("✅ Flow creation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Flow creation test failed: {e}")
        return False

def test_persona_integration():
    """Test persona integration with existing tools."""
    try:
        from .new_context_persona import NewContextPersona
        
        # Test that persona can be imported and has correct defaults
        class TestPersona(NewContextPersona):
            def __init__(self):
                pass
        
        persona = TestPersona()
        defaults = persona.defaults
        
        assert defaults.name == "NewContextPersona"
        assert "PocketFlow" in defaults.description
        assert "conversational" in defaults.system_prompt.lower()
        
        # Test intent analysis
        greeting = persona._analyze_message_intent("hello", [])
        assert greeting["type"] == "greeting"
        
        analysis = persona._analyze_message_intent("analyze my notebook: test.ipynb", [])
        assert analysis["type"] == "context_analysis"
        assert analysis["notebook_path"] == "test.ipynb"
        
        logger.info("✅ Persona integration test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Persona integration test failed: {e}")
        return False

def run_final_tests():
    """Run all final tests."""
    logger.info("🧪 Running final comprehensive tests...")
    
    tests = [
        ("PocketFlow Architecture", test_pocketflow_architecture),
        ("Context Nodes", test_context_nodes),
        ("Flow Creation", test_flow_creation),
        ("Persona Integration", test_persona_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n📊 Final Test Results:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Implementation is ready for use.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = run_final_tests()
    exit(0 if success else 1)