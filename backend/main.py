import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="Semantic Search System — powered by sentence-transformers + Groq")
console = Console()


def _get_engine():
    from src.search_engine import SearchEngine
    return SearchEngine()


def _get_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    doc_id: str = typer.Option(None, "--id", help="Custom doc ID (for plain text stdin)"),
):
    """Index a file, directory, or piped text into the vector store."""
    import sys
    from pathlib import Path

    engine = _get_engine()

    if not sys.stdin.isatty() and path == "-":
        text = sys.stdin.read()
        added = engine.index_text(text, doc_id=doc_id or "stdin")
        console.print(f"[green]Indexed {added} chunks from stdin[/green]")
        return

    p = Path(path)
    if not p.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    if p.is_dir():
        results = engine.index_directory(path)
        table = Table("Document", "Chunks Added")
        for doc_id, count in results.items():
            table.add_row(doc_id, str(count))
        console.print(table)
    else:
        added = engine.index_file(path)
        console.print(f"[green]Indexed {added} chunks from {p.name}[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
):
    """Semantic search — returns ranked document chunks."""
    engine = _get_engine()
    results = engine.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found. Have you indexed any documents?[/yellow]")
        return

    console.print(f"\n[bold]Results for:[/bold] {query}\n")
    for i, r in enumerate(results, 1):
        source = r.metadata.get("filename", r.metadata.get("doc_id", r.chunk_id))
        console.print(
            Panel(
                r.content,
                title=f"[{i}] {source}  |  score: {r.score:.4f}",
                border_style="blue",
            )
        )


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to answer"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Chunks to retrieve"),
    rerank: bool = typer.Option(False, "--rerank", help="Use LLM reranking"),
):
    """Ask a question — retrieves context and generates an answer via Groq."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(use_llm_rerank=rerank)
    with console.status("[bold green]Thinking...[/bold green]"):
        response = pipeline.ask(query, top_k=top_k)

    console.print(Panel(Markdown(response.answer), title="Answer", border_style="green"))
    console.print(f"\n[dim]Model: {response.model} | Sources used: {len(response.sources)}[/dim]\n")

    for i, src in enumerate(response.sources, 1):
        source = src.metadata.get("filename", src.metadata.get("doc_id", src.chunk_id))
        console.print(f"  [dim][{i}] {source} (score: {src.score:.4f})[/dim]")


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send directly to Groq"),
):
    """Chat directly with Groq — no document retrieval, just the LLM."""
    from src.llm_client import LLMClient

    llm = LLMClient()
    with console.status("[bold green]Thinking...[/bold green]"):
        reply = llm.chat(message)

    console.print(Panel(Markdown(reply), title="Groq", border_style="magenta"))


@app.command()
def stats():
    """Show index statistics."""
    engine = _get_engine()
    s = engine.stats()
    console.print(f"\n[bold]Index Stats[/bold]")
    console.print(f"  Total chunks : {s.total_chunks}")
    console.print(f"  Documents    : {len(s.documents)}")
    if s.documents:
        for doc in s.documents:
            console.print(f"    • {doc}")
    console.print()


@app.command()
def delete(
    doc_id: str = typer.Argument(..., help="Document ID to remove from the index"),
):
    """Remove a document from the index."""
    engine = _get_engine()
    engine.delete_document(doc_id)
    console.print(f"[green]Deleted document: {doc_id}[/green]")


if __name__ == "__main__":
    app()
