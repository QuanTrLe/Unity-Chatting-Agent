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
        
        # agent skills
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
        
        # capability and authentication
        capabilities = AgentCapabilities(streaming = True, push_notifications = True)
        
        # authentication and security schemes
        # client_credentials_flow = ClientCredentialsOAuthFlow(
        #     tokenUrl=f"http://localhost:8080/token", # Url to go to for client
        #     scopes = {'send_messages': 'agent tool to communicate and delegate tasks'}
        # )
        # oauth_flows = OAuthFlows(clientCredentials = client_credentials_flow)
        
        # oauth_scheme = OAuth2SecurityScheme(
        #     type = "oauth2",
        #     description = "OAuth 2.0 JWT token with 'send_messages' scope required",
        #     flows = oauth_flows,
        # )
        
        # card
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
            # securitySchemes = {
            #     "OAuth": oauth_scheme
            # },
            # security = [{
            #     "OAuth": ["send_messages"]
            # }],
        )
        
        client = httpx.AsyncClient()
        
        # authentication
        push_notification_config_store = InMemoryPushNotificationConfigStore()
        push_notification_config_sender = BasePushNotificationSender(
            httpx_client = client,
            config_store = push_notification_config_store
        )
        
        # request handling, bridging between the server and the client
        request_handler = DefaultRequestHandler(
            agent_executor = RoutingAgentExecutor(),
            task_store = InMemoryTaskStore(),
            push_config_store = push_notification_config_store,
            push_sender = push_notification_config_sender
        )
        
        # server using starlette app
        server = A2AStarletteApplication(
            agent_card = public_agent_card,
            http_handler = request_handler
        )
        
        # adding the authentication middleware layer
        app = server.build()
        # app.add_middleware(
        #     OAuthMiddleware, 
        #     agent_card = public_agent_card, 
        #     public_paths=['/.well-known/agent.json']
        # )
        
        uvicorn.run(app = app, host = host, port = port)
    
    except MissingAPIKeyError as error:
        logger.error(f"Error: {error}")
        exit(1)
        
    except Exception as error:
        logger.error(f"Error occurred during server startup: {error}")
        exit(1)


if __name__ == "__main__":
    main()
