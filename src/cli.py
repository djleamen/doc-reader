"""
Command-line interface for the RAG Document Q&A system.
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Union

from loguru import logger
from src.config import settings
from src.rag_engine import RAGEngine, ConversationalRAG

def setup_logging():
    '''Setup logging configuration.'''
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )

    # Also log to file
    log_file = Path(settings.logs_dir) / "rag_cli.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, rotation="10 MB", retention="7 days")


def add_documents_command(args):
    '''Add documents to the RAG system.'''
    logger.info(f"Adding documents to index: {args.index_name}")

    rag_engine = RAGEngine(args.index_name)

    # Check if files exist
    valid_files = []
    for file_path in args.files:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")

    if not valid_files:
        logger.error("No valid files to process")
        return

    try:
        rag_engine.add_documents(valid_files)
        logger.success(f"Successfully added {len(valid_files)} documents to index")
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        sys.exit(1)


def _print_query_result(result, args):
    '''Print query result with optional sources and metadata.'''
    print("\n" + "="*80)
    print("QUESTION:")
    print(args.question)
    print("\n" + "-"*80)
    print("ANSWER:")
    print(result.answer)

    if args.include_sources and result.source_documents:
        _print_source_documents(result, args)

    print("\n" + "-"*80)
    print("METADATA:")
    print(json.dumps(result.metadata, indent=2))
    print("="*80)


def _print_source_documents(result, args):
    '''Print source documents with scores.'''
    print("\n" + "-"*80)
    print("SOURCE DOCUMENTS:")
    for i, doc in enumerate(result.source_documents, 1):
        score_text = ""
        if args.include_scores and i <= len(result.confidence_scores):
            score_text = f" (Score: {result.confidence_scores[i-1]:.4f})"

        print(f"\nSource {i}{score_text}:")
        print(f"File: {doc.metadata.get('filename', 'Unknown')}")
        print(f"Chunk: {doc.metadata.get('chunk_id', 'Unknown')}")
        print("Content:")
        print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)


def _execute_query(rag_engine, args):
    '''Execute query based on engine type.'''
    if args.conversational:
        assert isinstance(rag_engine, ConversationalRAG)
        return rag_engine.conversational_query(
            question=args.question,
            k=args.top_k,
            include_sources=args.include_sources,
            include_scores=args.include_scores
        )
    else:
        return rag_engine.query(
            question=args.question,
            k=args.top_k,
            include_sources=args.include_sources,
            include_scores=args.include_scores
        )


def query_command(args):
    '''Query the RAG system.'''
    logger.info(f"Querying index: {args.index_name}")

    rag_engine: Union[RAGEngine, ConversationalRAG]
    if args.conversational:
        rag_engine = ConversationalRAG(args.index_name)
    else:
        rag_engine = RAGEngine(args.index_name)

    try:
        result = _execute_query(rag_engine, args)
        _print_query_result(result, args)
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        sys.exit(1)


def _handle_clear_command(rag_engine, args):
    '''Handle clear conversation command.'''
    if args.conversational:
        assert isinstance(rag_engine, ConversationalRAG)
        rag_engine.clear_conversation()
        print("💭 Conversation history cleared!")
        return True
    return False


def _process_interactive_question(rag_engine, question, args):
    '''Process a single question in interactive mode.'''
    print("🔍 Searching for answer...")

    if args.conversational:
        assert isinstance(rag_engine, ConversationalRAG)
        result = rag_engine.conversational_query(
            question=question,
            k=args.top_k,
            include_sources=False,
            include_scores=False
        )
    else:
        result = rag_engine.query(
            question=question,
            k=args.top_k,
            include_sources=False,
            include_scores=False
        )

    print(f"\n💬 Answer: {result.answer}")

    if args.verbose:
        print(f"\n📊 Retrieved {result.metadata.get('retrieval_count', 0)} relevant documents")


def interactive_mode(args):
    '''Interactive Q&A mode.'''
    logger.info(f"Starting interactive mode with index: {args.index_name}")

    if args.conversational:
        rag_engine = ConversationalRAG(args.index_name)
        print("🤖 Conversational RAG Mode - Your questions will be remembered!")
    else:
        rag_engine = RAGEngine(args.index_name)
        print("🤖 RAG Q&A Mode - Ask questions about your documents!")

    print("Type 'quit', 'exit', or 'bye' to exit.")
    print("Type 'clear' to clear conversation history (conversational mode only).")
    print("-" * 60)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if question.lower() == 'clear':
                _handle_clear_command(rag_engine, args)
                continue

            if not question:
                continue

            _process_interactive_question(rag_engine, question, args)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}")


def list_indexes_command(args):
    '''List available indexes.'''
    index_dir = Path(settings.index_dir)

    if not index_dir.exists():
        logger.info("No indexes directory found")
        return

    indexes = [d.name for d in index_dir.iterdir() if d.is_dir()]

    if indexes:
        print("Available indexes:")
        for index in indexes:
            print(f"  - {index}")
    else:
        print("No indexes found")


def stats_command(args):
    '''Show index statistics.'''
    rag_engine = RAGEngine(args.index_name)
    stats = rag_engine.get_index_stats()

    print(f"Index Statistics for: {args.index_name}")
    print("-" * 40)
    print(json.dumps(stats, indent=2))


def clear_index_command(args):
    '''Clear an index.'''
    if args.confirm or input(f"Are you sure you want to clear index '{args.index_name}'? (y/N): ").lower() == 'y':
        rag_engine = RAGEngine(args.index_name)
        rag_engine.clear_index()
        logger.success(f"Index '{args.index_name}' cleared")
    else:
        logger.info("Operation cancelled")


def main():
    '''Main CLI entry point.'''
    setup_logging()

    parser = argparse.ArgumentParser(description="RAG Document Q&A System CLI")
    parser.add_argument("--index-name", default="default", help="Name of the document index")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add documents command
    add_parser = subparsers.add_parser("add", help="Add documents to the index")
    add_parser.add_argument("files", nargs="+", help="Paths to documents to add")
    add_parser.set_defaults(func=add_documents_command)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the document index")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve")
    query_parser.add_argument("--include-sources", action="store_true", default=True, help="Include source documents")
    query_parser.add_argument("--include-scores", action="store_true", default=True, help="Include similarity scores")
    query_parser.add_argument("--conversational", action="store_true", help="Use conversational mode")
    query_parser.set_defaults(func=query_command)

    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive Q&A mode")
    interactive_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to retrieve")
    interactive_parser.add_argument("--conversational", action="store_true", help="Use conversational mode")
    interactive_parser.set_defaults(func=interactive_mode)

    # List indexes
    list_parser = subparsers.add_parser("list", help="List available indexes")
    list_parser.set_defaults(func=list_indexes_command)

    # Show stats
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.set_defaults(func=stats_command)

    # Clear index
    clear_parser = subparsers.add_parser("clear", help="Clear an index")
    clear_parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    clear_parser.set_defaults(func=clear_index_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
