"""Make one bounded live Tavily request after TAVILY_API_KEY is configured."""

from app.agent_tools import TavilySearch
from app.config import get_settings


def main() -> None:
    key = get_settings().tavily_api_key
    if not key:
        raise SystemExit("Set TAVILY_API_KEY in .env before the live smoke check")
    results = TavilySearch(key).search("official Python 3.13 release notes")
    if not results:
        raise SystemExit("Tavily returned no usable web results")
    for item in results:
        print(item.title, item.path)


if __name__ == "__main__":
    main()
