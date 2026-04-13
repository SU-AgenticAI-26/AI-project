#!/usr/bin/env python3
"""
Setup script for AI Research Project

This script helps you configure API keys and set up the project.
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from secrets_manager import setup_secrets_interactive

def main():
    print("🚀 AI Research Project Setup")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("research_apis.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Error: Python 3.11+ required")
        sys.exit(1)

    print("✅ Python version:", sys.version.split()[0])

    # Check if required packages are installed
    required_packages = [
        'streamlit', 'langchain', 'openai', 'requests',
        'arxiv', 'networkx', 'pyvis', 'faiss-cpu'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        if input("Continue anyway? (y/N): ").lower() != 'y':
            sys.exit(1)

    print("\n🔐 API Keys Configuration")
    print("-" * 30)
    setup_secrets_interactive()

    print("\n📁 Project Structure Check")
    print("-" * 30)

    # Create necessary directories
    dirs_to_create = [
        "rag_data/knowledge_maps",
        "rag_data/query_cache",
        "rag_data/sessions",
        "rag_data/vectorstore",
        "rag_data/documents",
        "collab_rag_data/vectorstore",
        "collab_rag_data/documents",
        "collab_rag_data/knowledge_maps",
        "collab_rag_data/cache",
        "collab_rag_data/sessions"
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")

    print("\n🎉 Setup Complete!")
    print("-" * 20)
    print("To run the integrated research app:")
    print("  streamlit run integrated_research_app.py")
    print("\nTo run the collaborative RAG:")
    print("  streamlit run collaborative_Rag.py")
    print("\nTo configure more API keys later:")
    print("  python secrets_manager.py")

if __name__ == "__main__":
    main()