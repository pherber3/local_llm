from tavily import TavilyClient
import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()


class WebSearcher:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key not found")
        self.client = TavilyClient(api_key=self.api_key)

    @lru_cache(maxsize=100)
    def search(self, query: str) -> list[dict]:
        """
        Perform web search with caching for efficiency.
        """
        response = self.client.search(query=query)
        return response['results']

    def format_results(self, results: list[dict]) -> str:
        """
        Format search results into readable text.
        """
        return "\n".join(f"- {r['title']}: {r['content']}" for r in results)
