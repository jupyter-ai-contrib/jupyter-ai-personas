"""
setup_rag_system.py

Setup script for the Python Data Science Handbook RAG system.
Run this script to initialize everything and verify it's working.
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'chromadb',
        'sentence-transformers', 
        'langchain',
        'nbformat',
        'gitpython'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def setup_rag_system():
    """Initialize the RAG system."""
    print("🚀 Setting up Python Data Science Handbook RAG system...")
    
    try:
        # Import and test the RAG system
        from rag_core import create_handbook_rag
        
        print("📚 Initializing RAG system (this may take 5-10 minutes on first run)...")
        rag = create_handbook_rag(force_rebuild=False)
        
        if rag:
            print("✅ RAG system initialized successfully!")
            
            # Test search functionality
            print("🔍 Testing search functionality...")
            results = rag.search("pandas dataframe groupby", k=2)
            
            if results:
                print(f"✅ Search test successful! Found {len(results)} results")
                print("📋 Sample result:")
                print(f"   Source: {results[0]['source']}")
                print(f"   Content: {results[0]['content'][:100]}...")
                return True
            else:
                print("❌ Search test failed - no results found")
                return False
        else:
            print("❌ RAG system initialization failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure rag_core.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def test_persona_integration():
    """Test the persona integration."""
    print("🧪 Testing persona integration...")
    
    try:
        from rag_integration_tool import test_rag_integration
        
        if test_rag_integration():
            print("✅ Persona integration test successful!")
            return True
        else:
            print("❌ Persona integration test failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure rag_integration_tool.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def get_system_status():
    """Get detailed system status."""
    print("📊 System Status:")
    
    # Check file structure
    files_to_check = [
        'rag_core.py',
        'rag_integration_tool.py', 
        'context_retrieval_persona.py',
        'file_reader_tool.py'
    ]
    
    print("\n📁 File Status:")
    for file in files_to_check:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
    
    # Check directories
    directories = [
        './PythonDataScienceHandbook',
        './vector_stores'
    ]
    
    print("\n📂 Directory Status:")
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            if directory == './PythonDataScienceHandbook':
                notebook_count = len(list(dir_path.glob('notebooks/*.ipynb')))
                print(f"✅ {directory} ({notebook_count} notebooks)")
            else:
                print(f"✅ {directory}")
        else:
            print(f"❌ {directory} - NOT FOUND")
    
    # Try to get RAG system stats
    try:
        from rag_integration_tool import create_simple_rag_tools
        rag_tool = create_simple_rag_tools()
        status = rag_tool.get_system_status()
        status_data = json.loads(status)
        
        print("\n🧠 RAG System Status:")
        print(f"   System Available: {status_data.get('rag_system_available', False)}")
        print(f"   Repository Exists: {status_data.get('repository_exists', False)}")
        print(f"   Vector Store Exists: {status_data.get('vector_store_exists', False)}")
        
        if status_data.get('total_chunks'):
            print(f"   Total Chunks: {status_data['total_chunks']}")
            
    except Exception as e:
        print(f"⚠️ Could not get RAG system status: {e}")


def main():
    """Main setup and test function."""
    print("🔧 Python Data Science Handbook RAG System Setup")
    print("=" * 50)
    
    # Step 1: Check dependencies
    print("\n1. Checking Dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and run again")
        return False
    
    # Step 2: Setup RAG system
    print("\n2. Setting up RAG System...")
    if not setup_rag_system():
        print("\n❌ RAG system setup failed")
        get_system_status()
        return False
    
    # Step 3: Test persona integration
    print("\n3. Testing Persona Integration...")
    if not test_persona_integration():
        print("\n⚠️ Persona integration test failed, but RAG core is working")
    
    # Step 4: Show system status
    print("\n4. Final System Status")
    get_system_status()
    
    print("\n🎉 Setup completed!")
    print("\n💡 Your RAG system is ready to use with the ContextRetrieverPersona")
    print("\n📖 Usage:")
    print("   1. Provide a prompt describing what you want to learn")
    print("   2. Include: notebook: /path/to/your/notebook.ipynb")
    print("   3. The system will analyze your notebook and find relevant handbook content")
    print("   4. You'll receive a comprehensive markdown report")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)