## SERVER SEARCH: has Tavily API for searching info ##


# Imports #

from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

# Initiating things #

load_dotenv()
mcp = FastMCP("Search")


# The functions and tools #

tool = TavilySearch(max_results = 3)

@mcp.tool()
async def search(search_query: str):
    search_result = tool.invoke({"query": search_query})
    return search_result


# Main #
if __name__ == "__main__":
    mcp.run(transport = "stdio")
