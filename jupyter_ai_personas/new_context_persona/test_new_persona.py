"""
Simple test for the new context persona implementation.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    try:
        from .new_context_persona import NewContextPersona
        from .pocketflow import Flow, Node, create_context_retrieval_flow
        
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_pocketflow_basic():
    """Test basic PocketFlow functionality."""
    try:
        from .pocketflow import create_context_retrieval_flow
        
        # Create mock tools for testing
        mock_notebook_tools = []
        mock_rag_tools = []
        mock_file_tools = []
        
        # Create flow
        flow = create_context_retrieval_flow(
            notebook_tools=mock_notebook_tools,
            rag_tools=mock_rag_tools,
            file_tools=mock_file_tools
        )
        
        # Test basic structure
        assert flow.name == "ContextRetrievalFlow"
        assert len(flow.nodes) == 3  # NotebookAnalysis, KnowledgeSearch, ReportGeneration
        
        logger.info("✅ PocketFlow basic test passed")
        return True
    except Exception as e:
        logger.error(f"❌ PocketFlow test failed: {e}")
        return False

def test_persona_defaults():
    """Test persona defaults and initialization."""
    try:
        from .new_context_persona import NewContextPersona
        
        # Test that we can create defaults (without full initialization)
        class MockPersona(NewContextPersona):
            def __init__(self):
                # Skip parent init to avoid dependencies
                pass
        
        mock_persona = MockPersona()
        defaults = mock_persona.defaults
        
        assert defaults.name == "NewContextPersona"
        assert "PocketFlow" in defaults.description
        assert "notebook analysis" in defaults.system_prompt.lower()
        
        logger.info("✅ Persona defaults test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Persona defaults test failed: {e}")
        return False

def test_intent_analysis():
    """Test intent analysis functionality."""
    try:
        from .new_context_persona import NewContextPersona
        
        # Create mock persona for testing
        class TestPersona(NewContextPersona):
            def __init__(self):
                # Skip parent init
                pass
        
        persona = TestPersona()
        
        # Test greeting detection
        greeting_result = persona._analyze_message_intent("hello", [])
        assert greeting_result["type"] == "greeting"
        
        # Test context analysis detection
        context_result = persona._analyze_message_intent("analyze notebook: test.ipynb", [])
        assert context_result["type"] == "context_analysis"
        assert context_result["notebook_path"] == "test.ipynb"
        
        # Test simple question detection
        question_result = persona._analyze_message_intent("what is pandas?", [])
        assert question_result["type"] == "simple_question"
        
        logger.info("✅ Intent analysis test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Intent analysis test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("🧪 Running New Context Persona tests...")
    
    tests = [
        ("Imports", test_imports),
        ("PocketFlow Basic", test_pocketflow_basic),
        ("Persona Defaults", test_persona_defaults),
        ("Intent Analysis", test_intent_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! New Context Persona is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)