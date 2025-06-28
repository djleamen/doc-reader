"""
Example usage of the RAG Document Q&A system.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_engine import RAGEngine, ConversationalRAG
from src.document_processor import DocumentProcessor
from src.utils import create_directory_structure
import tempfile


def create_sample_documents():
    """Create sample documents for demonstration."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Sample document 1: AI Research Paper
    doc1_content = """
    # Advances in Large Language Models: A Comprehensive Survey

    ## Abstract
    This paper presents a comprehensive survey of recent advances in large language models (LLMs), 
    covering their architecture, training methodologies, and applications. We analyze the evolution 
    from transformer-based models to modern architectures like GPT, BERT, and their variants.

    ## Introduction
    Large Language Models have revolutionized natural language processing tasks. These models, 
    trained on massive datasets, demonstrate remarkable capabilities in text generation, 
    comprehension, and reasoning. The field has seen rapid progress with models like GPT-3, 
    GPT-4, PaLM, and LaMDA showing human-level performance on various benchmarks.

    ## Methodology
    Our analysis covers models from 2017 to 2024, focusing on:
    1. Architecture improvements
    2. Training efficiency
    3. Scaling laws
    4. Fine-tuning techniques
    5. Evaluation metrics

    ## Key Findings
    1. Scaling model size generally improves performance but with diminishing returns
    2. Quality of training data is more important than quantity
    3. Instruction tuning significantly improves model alignment
    4. Retrieval-augmented generation (RAG) enhances factual accuracy
    5. Constitutional AI methods improve safety and reliability

    ## Applications
    LLMs have found applications in:
    - Code generation and debugging
    - Creative writing and content creation
    - Question answering systems
    - Language translation
    - Scientific research assistance
    - Educational tools

    ## Future Directions
    Future research should focus on:
    - Reducing computational requirements
    - Improving factual accuracy
    - Better alignment with human values
    - Multimodal capabilities
    - Efficient fine-tuning methods

    ## Conclusion
    Large Language Models represent a significant advancement in AI, with broad applications 
    across industries. However, challenges remain in terms of cost, reliability, and ethical 
    considerations. Continued research is essential for realizing their full potential.
    """
    
    # Sample document 2: Technical Manual
    doc2_content = """
    # RAG System Implementation Guide

    ## Overview
    Retrieval-Augmented Generation (RAG) systems combine information retrieval with 
    language generation to provide accurate, contextual responses based on a knowledge base.

    ## System Components

    ### Document Processor
    The document processor handles:
    - File format detection and parsing
    - Text extraction from PDF, DOCX, and other formats
    - Text cleaning and preprocessing
    - Chunking strategies for optimal retrieval

    ### Vector Store
    The vector store manages:
    - Document embeddings using models like text-embedding-ada-002
    - Similarity search algorithms (cosine similarity, dot product)
    - Index optimization for large document collections
    - Metadata storage for source tracking

    ### Language Model Integration
    Integration with LLMs includes:
    - Prompt engineering for optimal responses
    - Context window management
    - Temperature and parameter tuning
    - Response post-processing

    ## Implementation Best Practices

    ### Document Chunking
    Optimal chunking strategies:
    - Chunk size: 500-1500 tokens
    - Overlap: 100-300 tokens
    - Preserve semantic boundaries
    - Include metadata for source tracking

    ### Embedding Strategy
    Choose embeddings based on:
    - Domain specificity
    - Language requirements
    - Performance vs. cost tradeoffs
    - Update frequency needs

    ### Retrieval Optimization
    Improve retrieval through:
    - Query expansion techniques
    - Re-ranking algorithms
    - Hybrid search approaches
    - Result diversity optimization

    ## Performance Tuning

    ### Latency Optimization
    - Cache frequently accessed embeddings
    - Use approximate nearest neighbor search
    - Implement result pagination
    - Optimize vector database configuration

    ### Accuracy Improvements
    - Fine-tune retrieval parameters
    - Implement relevance feedback
    - Use multiple retrieval strategies
    - Validate responses against sources

    ## Troubleshooting

    ### Common Issues
    1. Poor retrieval relevance
       - Solution: Adjust chunk size and overlap
       - Check embedding model compatibility
    
    2. Slow query response
       - Solution: Optimize vector database
       - Implement caching strategies
    
    3. Inconsistent answers
       - Solution: Improve prompt engineering
       - Add context validation

    ## Monitoring and Evaluation
    Track key metrics:
    - Query response time
    - Retrieval accuracy
    - User satisfaction scores
    - System resource usage
    """
    
    # Sample document 3: Business Report
    doc3_content = """
    # Q4 2024 AI Technology Market Analysis

    ## Executive Summary
    The AI technology market experienced unprecedented growth in Q4 2024, driven primarily 
    by enterprise adoption of large language models and generative AI solutions. Market 
    capitalization reached $2.3 trillion, representing 45% year-over-year growth.

    ## Market Segments

    ### Large Language Models
    The LLM segment dominated with 40% market share:
    - OpenAI maintained leadership with GPT-4 and GPT-5
    - Google's Gemini gained significant enterprise traction
    - Anthropic's Claude showed strong performance in safety benchmarks
    - Meta's Llama models drove open-source adoption

    ### Enterprise AI Solutions
    Enterprise solutions grew 60% year-over-year:
    - Document processing and analysis: $450B market
    - Customer service automation: $230B market
    - Code generation tools: $180B market
    - Content creation platforms: $120B market

    ### Infrastructure and Tools
    AI infrastructure investments increased substantially:
    - Vector databases: 300% growth
    - GPU computing resources: 200% growth
    - MLOps platforms: 150% growth
    - Edge AI solutions: 180% growth

    ## Regional Analysis

    ### North America
    - Market share: 45%
    - Growth rate: 42%
    - Key players: OpenAI, Google, Microsoft, Meta
    - Focus areas: Enterprise AI, research, regulation

    ### Europe
    - Market share: 25%
    - Growth rate: 38%
    - Key initiatives: AI Act compliance, sovereign AI
    - Leading companies: Stability AI, DeepMind, Mistral

    ### Asia-Pacific
    - Market share: 30%
    - Growth rate: 55%
    - Major players: Alibaba, Baidu, ByteDance, Anthropic
    - Growth drivers: Manufacturing automation, smart cities

    ## Industry Applications

    ### Healthcare
    AI adoption accelerated in:
    - Medical imaging analysis
    - Drug discovery and development
    - Clinical decision support
    - Administrative automation

    ### Financial Services
    Key applications include:
    - Fraud detection and prevention
    - Algorithmic trading
    - Risk assessment
    - Customer service chatbots

    ### Education
    Transformative applications:
    - Personalized learning platforms
    - Automated grading and feedback
    - Content generation for curricula
    - Language learning assistance

    ## Technology Trends

    ### Multimodal AI
    Integration of text, image, and audio processing became mainstream:
    - GPT-4V and Gemini Pro Vision led adoption
    - Real-time video analysis capabilities
    - Enhanced accessibility features

    ### Retrieval-Augmented Generation
    RAG systems gained widespread enterprise adoption:
    - 400% increase in RAG implementations
    - Integration with existing knowledge bases
    - Improved accuracy and reduced hallucinations

    ### Edge AI Computing
    Deployment of AI at the edge expanded:
    - Mobile AI processing capabilities
    - IoT device integration
    - Reduced latency requirements

    ## Challenges and Opportunities

    ### Regulatory Landscape
    - EU AI Act implementation
    - US executive orders on AI safety
    - China's AI governance framework
    - Industry self-regulation initiatives

    ### Technical Challenges
    - Model interpretability and explainability
    - Computational efficiency and costs
    - Data privacy and security
    - Bias mitigation and fairness

    ### Market Opportunities
    - Small and medium enterprise adoption
    - Vertical-specific AI solutions
    - AI-human collaboration tools
    - Sustainable AI practices

    ## 2025 Predictions
    Expected developments for 2025:
    1. AGI prototypes reach limited deployment
    2. AI agents become mainstream in enterprises
    3. Multimodal AI integrates across all major platforms
    4. Edge AI capabilities match cloud performance
    5. Regulatory frameworks stabilize globally

    ## Conclusion
    The AI technology market continues its rapid expansion with increasing enterprise 
    adoption and technological advancement. Success in 2025 will depend on balancing 
    innovation with responsibility, scalability with efficiency, and capability with safety.
    """
    
    # Write documents to temporary files
    docs = [
        (doc1_content, "ai_research_paper.md"),
        (doc2_content, "rag_implementation_guide.md"),
        (doc3_content, "q4_2024_ai_market_analysis.md")
    ]
    
    for content, filename in docs:
        doc_path = temp_dir / filename
        doc_path.write_text(content)
    
    return temp_dir


def demonstrate_basic_rag():
    """Demonstrate basic RAG functionality."""
    print("🚀 Basic RAG Demonstration")
    print("=" * 50)
    
    # Create sample documents
    docs_dir = create_sample_documents()
    print(f"📁 Created sample documents in: {docs_dir}")
    
    # Initialize RAG engine
    rag = RAGEngine(index_name="demo")
    
    # Add documents
    doc_files = list(docs_dir.glob("*.md"))
    print(f"📚 Adding {len(doc_files)} documents to index...")
    rag.add_documents([str(f) for f in doc_files])
    
    # Sample queries
    queries = [
        "What are the key findings about large language models?",
        "How should I implement document chunking in a RAG system?",
        "What was the market size for AI technology in Q4 2024?",
        "What are the best practices for retrieval optimization?",
        "Which companies are leading in the LLM market?"
    ]
    
    print("\n🔍 Running sample queries:")
    print("-" * 30)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = rag.query(query, k=3, include_sources=True, include_scores=True)
        
        print(f"Answer: {result.answer[:200]}...")
        print(f"Sources: {len(result.source_documents)} documents")
        
        if result.confidence_scores:
            avg_score = sum(result.confidence_scores) / len(result.confidence_scores)
            print(f"Avg Confidence: {avg_score:.3f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(docs_dir)
    rag.clear_index()
    
    print("\n✅ Basic RAG demonstration completed!")


def demonstrate_conversational_rag():
    """Demonstrate conversational RAG functionality."""
    print("\n🗣️ Conversational RAG Demonstration")
    print("=" * 50)
    
    # Create sample documents
    docs_dir = create_sample_documents()
    
    # Initialize conversational RAG
    conv_rag = ConversationalRAG(index_name="conv_demo")
    
    # Add documents
    doc_files = list(docs_dir.glob("*.md"))
    conv_rag.add_documents([str(f) for f in doc_files])
    
    # Conversational queries that build on each other
    conversation = [
        "What are large language models?",
        "How do they relate to RAG systems?",
        "What are the implementation challenges mentioned?",
        "How did the market perform for these technologies?",
        "What are the predictions for next year?"
    ]
    
    print("\n💬 Running conversational queries:")
    print("-" * 35)
    
    for i, query in enumerate(conversation, 1):
        print(f"\nTurn {i}: {query}")
        result = conv_rag.conversational_query(query, k=2, include_sources=False)
        print(f"Answer: {result.answer[:150]}...")
    
    # Cleanup
    import shutil
    shutil.rmtree(docs_dir)
    conv_rag.clear_index()
    
    print("\n✅ Conversational RAG demonstration completed!")


def demonstrate_document_processing():
    """Demonstrate document processing capabilities."""
    print("\n📄 Document Processing Demonstration")
    print("=" * 50)
    
    # Create sample documents
    docs_dir = create_sample_documents()
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    print("\n🔧 Processing documents:")
    print("-" * 25)
    
    for doc_file in docs_dir.glob("*.md"):
        print(f"\nProcessing: {doc_file.name}")
        
        # Process document
        documents = processor.process_document(str(doc_file))
        
        print(f"  📊 Created {len(documents)} chunks")
        print(f"  📏 First chunk size: {len(documents[0].page_content)} chars")
        print(f"  🏷️ Metadata keys: {list(documents[0].metadata.keys())}")
        print(f"  📝 Preview: {documents[0].page_content[:100]}...")
    
    # Cleanup
    import shutil
    shutil.rmtree(docs_dir)
    
    print("\n✅ Document processing demonstration completed!")


def main():
    """Run all demonstrations."""
    print("🎯 RAG Document Q&A System - Example Usage")
    print("=" * 60)
    
    try:
        # Ensure directories exist
        create_directory_structure()
        
        # Run demonstrations
        demonstrate_document_processing()
        demonstrate_basic_rag()
        demonstrate_conversational_rag()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Set up your .env file with OpenAI API key")
        print("2. Add your own documents to the 'documents' folder")
        print("3. Start the web interface: streamlit run src/streamlit_app.py")
        print("4. Or use the CLI: python -m src.cli --help")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure you have installed all dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
