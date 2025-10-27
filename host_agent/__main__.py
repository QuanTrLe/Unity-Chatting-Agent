## FOR USING WITH PUSH NOTIF AND AUTHENTICATION ###

# ASGI web server implementation
import uvicorn
import logging
import httpx

import os
from dotenv import load_dotenv
import click

# lightweight ASGI framework/toolkit to build async web
from a2a.server.apps import A2AStarletteApplication
# for handling the invoke and conencting to our executing agent
from a2a.server.request_handlers import DefaultRequestHandler
# memory so that we can enqueue messages and tasks
from a2a.server.tasks import (
    InMemoryTaskStore, 
    InMemoryPushNotificationConfigStore, 
    BasePushNotificationSender
)

# types we use to declare the agent and its capabilities
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    OAuth2SecurityScheme,
    OAuthFlows,
    ClientCredentialsOAuthFlow
)
from bearer_middleware import OAuthMiddleware

# the executor
from agent_executor import RoutingAgentExecutor

# --- NEW IMPORTS ---
from starlette.responses import JSONResponse
from starlette.requests import Request
# --------------------


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10004)
def main(host, port):
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable is not set")
        
        # (Your existing AgentSkill, Capabilities, Card, etc. code remains here...)
        # ...
        skill = AgentSkill(
            id = "send_messages",
            name = "Send Messages",
            description = "Delegate and asks agents specialized for movie, points of interest, and weather tasks",
            tags = ["delegating", "send messages", "tasks", "points of interest"],
            examples = [
                "What are some comedy horror movies that came out after 2010", 
                "What is the weather of Bethlehem, PA in USA?", 
                "What restaurants are near Lehigh University, PA, in USA?"
            ]
        )
        capabilities = AgentCapabilities(streaming = True, push_notifications = True)
        public_agent_card = AgentCard(
            name = "Routing Agent",
            description = """
            An agent that delegates tasks, choosing which remote seller agent to ask about movies, 
            points of interest, or weather inquiries
            """,
            url = f"http://{host}:{port}/",
            version = "1.0.0",
            defaultInputModes = ['text', 'text/plain'],
            defaultOutputModes = ['text', 'text/plain'],
            capabilities = capabilities,
            skills = [skill],
        )
        # ...
        
        client = httpx.AsyncClient()
        
        # authentication
        push_notification_config_store = InMemoryPushNotificationConfigStore()
        push_notification_config_sender = BasePushNotificationSender(
            httpx_client = client,
            config_store = push_notification_config_store
        )
        
        # --- MODIFICATION: Get the agent instance ---
        # We need the agent itself to call its .stream() method
        agent_executor = RoutingAgentExecutor()
        routing_agent = agent_executor.agent # Get the agent from the executor
        # ---------------------------------------------
        
        # request handling, bridging between the server and the client
        request_handler = DefaultRequestHandler(
            agent_executor = agent_executor, # Pass the executor
            task_store = InMemoryTaskStore(),
            push_config_store = push_notification_config_store,
            push_sender = push_notification_config_sender
        )
        
        # server using starlette app
        server = A2AStarletteApplication(
            agent_card = public_agent_card,
            http_handler = request_handler
        )
        
        
        # --- NEW: Define the simple chat endpoint ---
        async def simple_chat_endpoint(request: Request):
            try:
                data = await request.json()
                query = data.get("query")
                # Your routing_agent.stream() needs a session_id
                session_id = data.get("session_id", "default-unity-session") 
                
                if not query:
                    return JSONResponse({"error": "No query provided"}, status_code=400)

                final_response = "Agent did not provide a final response."
                
                # Call your agent's stream method directly
                async for finished, text in routing_agent.stream(query, session_id):
                    if finished:
                        final_response = text
                        break # We got the final answer, so we stop
                
                # Return the final answer as simple JSON
                return JSONResponse({"response": final_response})
            
            except Exception as e:
                logger.error(f"Error in simple_chat_endpoint: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        # -------------------------------------------

        
        # adding the authentication middleware layer
        app = server.build()
        
        # --- NEW: Add the new route to the app ---
        # This makes http://localhost:10004/simple_chat available
        app.add_route("/simple_chat", simple_chat_endpoint, methods=["POST"])
        # -----------------------------------------
        
        # (Your commented-out app.add_middleware... code can stay here)
        
        uvicorn.run(app = app, host = host, port = port)
    
    except MissingAPIKeyError as error:
        logger.error(f"Error: {error}")
        exit(1)
        
    except Exception as error:
        logger.error(f"Error occurred during server startup: {error}")
        exit(1)


if __name__ == "__main__":
    main()