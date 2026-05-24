# tools/lightweight_search.py
# Custom search tool that aggressively truncates results to stay within Groq free tier TPM limits

import os
import requests
from crewai.tools import BaseTool


class LightweightSearchTool(BaseTool):
    name: str = "Indian Finance Search Tool"
    description: str = (
        "Searches latest Indian mutual fund and market information "
        "and returns concise summaries only. Use for specific factual queries."
    )

    def _run(self, query: str) -> str:
        try:
            url = "https://google.serper.dev/search"
            payload = { "q": query, "num": 2 }
            headers = {
                "X-API-KEY": os.getenv("SERPER_API_KEY"),
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            data = response.json()
            organic = data.get("organic", [])

            results = []
            for item in organic[:2]:
                title   = item.get("title", "")[:60]
                snippet = item.get("snippet", "")[:120]
                results.append(f"- {title}: {snippet}")

            final_result = "\n".join(results)

            # Hard truncation — never exceed 400 chars
            return final_result[:400]

        except Exception as e:
            return f"Search failed: {str(e)[:100]}"
