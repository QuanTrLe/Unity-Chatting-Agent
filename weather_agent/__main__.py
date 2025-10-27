import asyncio
import os
import sys

from contextlib import asynccontextmanager
from typing import Any

import click
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import WeatherAgentExecutor
from weather_agent import WeatherAgent

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv(override=True)


SERVER_CONFIGS = {
    # uses the NWS (National Weather Service), that gives forecast and alerts but only in USA
    'nws': {
        'command': 'node',
        'args': ['../mcp_servers/nws_weather_mcp_server/build/index.js'],
        'transport': 'stdio',
    },
    # uses the Tavily search tool to get the coordinates and state codes that NWS asks for
    'tavily': {
        'command': 'python',
        'args': ['../mcp_servers/tavily_mcp_server.py'],
        'transport': 'stdio',
    }
}

app_context: dict[str, Any] = {}

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 10002
DEFAULT_LOG_LEVEL = 'info'


@asynccontextmanager
async def app_lifespan(context: dict[str, Any]):
    """Manages the lifecycle of shared resources like the MCP client and tools."""
    
    # make the server and get the tools from the servers we hooked up with
    mcp_client_instance: MultiServerMCPClient | None = None
    
    # getting tools and assigning them into the context
    try:
        mcp_client_instance = MultiServerMCPClient(SERVER_CONFIGS)
        mcp_tools = await mcp_client_instance.get_tools()
        context['mcp_tools'] = mcp_tools
        for tool in mcp_tools: print(f"MCP TOOL: {tool}")
        
        tool_count = len(mcp_tools) if mcp_tools else 0
        print(
            f'Lifespan: MCP Tools preloaded successfully ({tool_count} tools found).'
        )
        yield
    
    except Exception as error:
        print(f'Lifespan: Error during initialization: {error}', file=sys.stderr)
        raise

    # shutting down the client after we're done to free up resources
    finally:
        print('Lifespan: Shutting down MCP client...')
        
        if (mcp_client_instance): # if we do indeed have a client going on
            
            # if the instance does have an aexit like the original code
            if hasattr(mcp_client_instance, '__aexit__'):
                try:
                    print(
                        f'Lifespan: Calling __aexit__ on {type(mcp_client_instance).__name__} instance...'
                    )
                    await mcp_client_instance.__aexit__(None, None, None)
                    print(
                        'Lifespan: MCP Client resources released via __aexit__.'
                    )
                
                except Exception as error:
                    print(
                        f'Lifespan: Error during MCP client __aexit__: {error}',
                        file = sys.stderr,
                    )
            
            # clean method doesnt exist anymore, unexpected if only the context manager usage changed
            # could resource leak
            else:
                print(
                    f'Lifespan: CRITICAL - {type(mcp_client_instance).__name__} instance does not have __aexit__ method for cleanup. Resource leak possible.',
                    file=sys.stderr,
                )
        
        # if we dont have any client instance going on
        else:
            # MultiServerMCPClient() constructor likely failed or was not reached.
            print(
                'Lifespan: MCP Client instance was not created, no shutdown attempt via __aexit__.'
            )
        
        # Clear the application context as in the original code.
        print('Lifespan: Clearing application context.')
        context.clear()


def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    """Command Line Interface to start the Airbnb Agent server."""
    
    # make sure that api key is ok, dont need to if using vertexai
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv(
        'GOOGLE_API_KEY'
    ):
        raise ValueError(
            'GOOGLE_API_KEY environment variable not set and '
            'GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
        )

    async def run_server_async():
        async with app_lifespan(app_context):
            # making sure we have mcp tools loaded
            if not app_context.get('mcp_tools'):
                print(
                    'Warning: MCP tools were not loaded. Agent may not function correctly.',
                    file = sys.stderr,
                )
            
            # initialize Agent kit with preloaded tools
            weather_agent_executor = WeatherAgentExecutor(mcp_tools = app_context.get('mcp_tools', []))
            request_handler = DefaultRequestHandler(
                agent_executor = weather_agent_executor,
                task_store = InMemoryTaskStore()
            )
            a2a_server = A2AStarletteApplication(
                agent_card = get_agent_card(host, port),
                http_handler = request_handler
            )
            
            # get the ASGI app from the A2A server instance
            asgi_app = a2a_server.build()

            config = uvicorn.Config(
                app = asgi_app,
                host = host,
                port = port,
                log_level = log_level.lower(),
                lifespan = 'auto',
            )
            uvicorn_server = uvicorn.Server(config)

            print(
                f'Starting Uvicorn server at http://{host}:{port} with log-level {log_level}...'
            )
            
            # Keep online until we get stopped by keyboard
            try:
                await uvicorn_server.serve()
            except KeyboardInterrupt:
                print('Server shutdown requested (KeyboardInterrupt).')
            finally:
                print('Uvicorn server has stopped.')

    # run the server
    try:
        asyncio.run(run_server_async())
    
    # error cases
    except RuntimeError as error:
        if 'cannot be called from a running event loop' in str(error):
            print(
                'Critical Error: Attempted to nest asyncio.run(). This should have been prevented.',
                file = sys.stderr,
            )
        else:
            print(f'RuntimeError in main: {error}', file = sys.stderr)
        sys.exit(1)
    
    except Exception as error:
        print(f'An unexpected error occurred in main: {error}', file = sys.stderr)
        sys.exit(1)
        

def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Weather Agent."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    agent_skills = [
        AgentSkill(
            id = 'current_weather_search',
            name = 'Search current weather information',
            description = """Helps with the current weather information search using MCP tools""",
            tags = ['current weather', 'search'],
            examples = [ 
                'Is it snowing in New York, USA?',
                'What is the weather like right now in Bethlehem, PA in USA?'
            ],
        ),
        AgentSkill(
            id = 'weather_forecast_search',
            name = 'Search weather forecast information',
            description = """Helps with the weather forecast search using MCP tools""",
            tags = ['weather forecast', 'search'],
            examples = [ 
                'What will the weather be like tomorrow in NewYork, USA?',
                'Will there be any rainy day next week in Bethlehem, PA in USA?'
            ],
        )
    ]
    return AgentCard(
        name = 'Weather Agent',
        description = 'Helps with searching weather information',
        url = f'http://{host}:{port}/',
        version = '1.0.0',
        defaultInputModes = WeatherAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes = WeatherAgent.SUPPORTED_CONTENT_TYPES,
        capabilities = capabilities,
        skills = agent_skills,
    )


@click.command()
@click.option(
    '--host',
    'host',
    default=DEFAULT_HOST,
    help='Hostname to bind the server to.',
)
@click.option(
    '--port',
    'port',
    default=DEFAULT_PORT,
    type=int,
    help='Port to bind the server to.',
)
@click.option(
    '--log-level',
    'log_level',
    default=DEFAULT_LOG_LEVEL,
    help='Uvicorn log level.',
)
def cli(host: str, port: int, log_level: str):
    main(host, port, log_level)


if __name__ == '__main__':
    main()