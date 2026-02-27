import os

from tavily import TavilyClient

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in environment")
        _client = TavilyClient(api_key=api_key)
    return _client


# Domain lists for filtered searches
_DOCS_DOMAINS = [
    "docs.aws.amazon.com",
    "cloud.google.com",
    "learn.microsoft.com",
    "docs.confluent.io",
    "redis.io",
    "www.mongodb.com",
    "docs.docker.com",
    "kubernetes.io",
]

_CASE_STUDY_DOMAINS = [
    "engineering.atspotify.com",
    "netflixtechblog.com",
    "aws.amazon.com/blogs",
    "cloud.google.com/blog",
    "eng.uber.com",
    "blog.cloudflare.com",
    "github.blog",
    "medium.com",
    "dev.to",
]


def run_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: list[str] | None = None,
) -> list[dict]:
    """Run a Tavily search. Returns [{url, title, content}, ...]."""
    client = _get_client()
    kwargs: dict = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    response = client.search(**kwargs)
    results = response.get("results", [])
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),  # full content, no truncation
        }
        for r in results
    ]


def search_for_docs(query: str, max_results: int = 5) -> list[dict]:
    """Search official documentation sites."""
    return run_search(query, max_results=max_results, include_domains=_DOCS_DOMAINS)


def search_for_case_studies(query: str, max_results: int = 5) -> list[dict]:
    """Search engineering blogs and case studies."""
    return run_search(query, max_results=max_results, include_domains=_CASE_STUDY_DOMAINS)


def search_for_failures(query: str, max_results: int = 5) -> list[dict]:
    """Search for failure cases, outage reports, post-mortems."""
    return run_search(f"{query} outage OR failure OR post-mortem", max_results=max_results)


def format_results_for_prompt(results: list[dict]) -> str:
    """Format search results into a readable block for prompt injection."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['content']}")
        lines.append("")
    return "\n".join(lines)
