import asyncio
import json
import uuid

from typing import Any

from collections.abc import AsyncIterable
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext


load_dotenv()

# supervising and asking these agents
remote_agents_urls = {
    'water_measurement_agent': 'http://localhost:10001',
    'weather_agent': 'http://localhost:10002',
}


# Convert parts to text
def convert_part(part: Part, tool_context: ToolContext):
    """Convert a part to text. Only text parts are supported."""
    if part.type == 'text':
        return part.text

    return f'Unknown type: {part.type}'


def convert_parts(parts: list[Part], tool_context: ToolContext):
    """Convert parts to text."""
    converted_result = []
    for p in parts:
        converted_result.append(convert_part(p, tool_context))
    return converted_result


# Creates the payload with all the shingles in the form of dict 'message' key
def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None, message_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'type': 'text', 'text': text}],
        },
    }

    if message_id:
        payload['message']['messageId'] = message_id
        
    if task_id:
        payload['message']['taskId'] = task_id

    if context_id:
        payload['message']['contextId'] = context_id
    return payload


class RoutingAgent:
    """The Routing Agent
    
    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """
    
    def __init__(self, task_callback: TaskUpdateCallback | None = None):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''
        
        self._user_id = ''
        self._agent = None
        self._runner = None
    
    
    # Get the cards themselves from a given list and initialize agents as RemoteConnections
    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        """Asynchronous part of initialization."""
        # single httpx.AsyncClient for all card resolutions for efficiency
        # remote addresses are in static list above and made in initialize
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses: # iterate over each agent address
                
                 # card resolver for specific address
                card_resolver = A2ACardResolver(client, address)
                
                try:
                    # card from initialized resolver, then connection to agent
                    card = (
                        await card_resolver.get_agent_card()
                    )
                    remote_connection = RemoteAgentConnections(
                        agent_card = card, agent_url = address
                    )
                    
                    # fill the empty dictionary from init with the card and the connection
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card # for AgentCards specifically
                    
                # simple failure cases
                except httpx.ConnectError as e:
                    print(
                        f'ERROR: Failed to get agent card from {address}: {e}'
                    )
                except Exception as e:  # Catch other potential errors
                    print(
                        f'ERROR: Failed to initialize connection for {address}: {e}'
                    )

        # Populate self.agents using the logic from original __init__ (via list_remote_agents)
        agent_info = []
        for agent_detail_dict in self.list_remote_agents(): # card name and card info
            agent_info.append(json.dumps(agent_detail_dict))
        self.agents = '\n'.join(agent_info) # string description of ourselves
    
    
    @classmethod
    async def create( # instantiation func for the class itself
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None
    ) -> 'RoutingAgent':
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance
    
    
    # Creates the agent instance
    def create_agent(self) -> Agent:
        """Create an instance of the RoutingAgent."""
        model_id = 'gemini-2.0-flash-001'
        print(f'Using hardcoded model: {model_id}')
        
        return Agent(
            name = 'Routing_agent',
            model = model_id,
            instruction = self.root_instruction,
            before_model_callback = self.before_model_callback,
            description = (
                'This Routing Agent orchestrates the decomposition of the user asking for weather forecast or airbnb accommodation.'
            ),
            tools = [self.send_message]
        )
    
    
    # Prompt to guide the bot to focus on sending messages and delegating tasks
    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate the root instruction for the RoutingAgent."""
        current_agent = self.check_active_agent(context)
        
        return f"""
        **Role:** You are an expert Routing Delegator. Your primary function is to accurately 
        delegate user inquiries regarding the weather, water measurement, or flood monitoring to the 
        appropriate specialized remote agents.
        
        
        **Core Directives:**
        * **Task Delegation:** Utilize the `send_message` function to assign actionable tasks to 
        remote agents.
        
        * **Contextual Awareness for Remote Agents:** If a remote agent repeatedly requests user 
        confirmation, assume it lacks access to the full conversation history. In such cases, enrich 
        the task description with all necessary contextual information relevant to that specific agent.
        
        * **Autonomous Agent Engagement:** Never seek user permission before engaging with remote 
        agents. If multiple agents are required to fulfill a request, connect with them directly 
        without requesting user preference or confirmation.
        
        * **Transparent Communication:** Always present the complete and detailed response from the 
        remote agent to the user.
        
        * **User Confirmation Relay:** If a remote agent asks for confirmation, and the user has not 
        already provided it, relay this confirmation request to the user.
        
        * **Focused Information Sharing:** Provide remote agents with only relevant contextual 
        information. Avoid extraneous details.
        
        * **No Redundant Confirmations:** Do not ask remote agents for confirmation of information 
        or actions.
        
        * **Tool Reliance:** Strictly rely on available tools to address user requests. Do not 
        generate responses based on assumptions. If information is insufficient, request 
        clarification from the user.
        
        * **Prioritize Recent Interaction:** Focus primarily on the most recent parts of the 
        conversation when processing requests.
        
        * **Fully complete answers:** Make sure to fully answer the user's questions in the query when
        you generate the final response. 
        
        * **Starting new tasks:** Whenever the user asks a new question, always make a new task. In this case, 
        call `send_message` with `new_task=True` to signal that a new task should be created.


        **Agent Roster:**

        * Available Agents: `{self.agents}`
        * Currently Active Seller Agent: `{current_agent['active_agent']}`
        """
        
        
    # Basically make sure that there are still active agent in the state of context
    def check_active_agent(self, context: ReadonlyContext):
        """Helper function for before model callback, checking and return active agents"""
        state = context.state
        
        if (
            'session_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'active_agent' in state
        ):
            return {'active_agent': f'{state["active_agent"]}'}
        return {'active_agent': 'None'}
    
    
    # Making sure that the session is marked as active with an id for it before doing callback
    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        """Marking a session as active recently, helps decision to be more relevant"""
        state = callback_context.state
        
        if 'session_active' not in state or not state['session_active']: # if not active
            if 'session_id' not in state: # if this is a new session
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True # mark as active
    
    
    # From the cards we have get name and description of each one
    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f'Found agent card: {card.model_dump(exclude_none=True)}')
            print('=' * 100)
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    
    # The only tool the model uses to send messages to the remote seller agents
    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext, new_task: bool = False):
        """Sends a task to remote seller agent.

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.
            new_task: Set to True if this is a new task, even for the same agent.

        Yields:
            A dictionary of JSON data.
        """
        # making sure we have it in the registry
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f'Agent {agent_name} not found')
        
        # agent name from tool invocation context
        state = tool_context.state
        
        # if new task or asking new agent, reset task and context Id
        if new_task or ('active_agent' in state and state['active_agent'] != agent_name):
            state['task_id'] = None
            state['context_id'] = None
        
        state['active_agent'] = agent_name
        client = self.remote_agent_connections[agent_name] # get client from remote connection dictionary

        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        
        # Only getting the task and context id, not generating them since already done by seller agent executors
        task_id = state.get('task_id')
        context_id = state.get('context_id')

        message_id = ''
        metadata = {}
        if 'input_message_metadata' in state:
            metadata.update(**state['input_message_metadata']) # update metadata
            if 'message_id' in state['input_message_metadata']:
                message_id = state['input_message_metadata']['message_id'] # getting message id 
        if not message_id:
            message_id = str(uuid.uuid4())

        payload = create_send_message_payload(
            text = task, 
            task_id = task_id, 
            context_id = context_id, 
            message_id = message_id
        )

        # send the actual message request itself
        message_request = SendMessageRequest(
            id = message_id, params = MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(
            message_request = message_request
        )
        print( # parse the thing out
            'send_response',
            send_response.model_dump_json(exclude_none=True, indent=2),
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print('received non-success response. Aborting get task ')
            return None

        if not isinstance(send_response.root.result, Task):
            print('received non-task response. Aborting get task ')
            return None

        # after a successful response, store the task and context Id
        # so can be used in subsequent requests.
        if send_response.root.result.id:
            state['task_id'] = send_response.root.result.id
        if send_response.root.result.context_id:
            state['context_id'] = send_response.root.result.context_id

        return send_response.root.result
    
    
    async def stream(self, query, session_id) -> AsyncIterable[tuple[bool, str]]:
        # safeguard on None fields and the initialization order
        # set agent and runner
        if not self._agent:
            self._agent = self.create_agent()
        if not self._runner:
            self._user_id = 'routing_agent'
            self._runner = Runner( # class used to run agents
                app_name = self._agent.name,
                agent = self._agent,
                artifact_service = InMemoryArtifactService(),
                session_service = InMemorySessionService(),
                memory_service = InMemoryMemoryService()
            )
        
        # get or initialize the session itself
        session = await self._runner.session_service.get_session(
            app_name = self._agent.name,
            user_id = self._user_id,
            session_id = session_id
        )
        
        content = types.Content(
            role='user', parts=[types.Part.from_text(text=query)]
        )
        
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name = self._agent.name,
                user_id = self._user_id,
                state = {},
                session_id = session_id
            )
        
        # the stremaing in the session itself
        async for event in self._runner.run_async(
            user_id = self._user_id, session_id = session.id, new_message = content
        ):
            if event.is_final_response():
                yield(
                    True,
                    '\n'.join([p.text for p in event.content.parts if p.text])
                )
            else:
                yield (False, 'working...')
    
    
# Function that initializes the routing agent itself along with the list, basically main 
def get_initialized_routing_agent() -> 'RoutingAgent':
    """Synchronously creates and initializes the RoutingAgent."""

    async def _async_main() -> 'RoutingAgent':
        # Create the routing agent instance using the .create() classmethod
        routing_agent_instance = await RoutingAgent.create(
            remote_agent_addresses=[remote_agents_urls[n] for n in remote_agents_urls]
        )
        return routing_agent_instance

    try:
        # Run the async initialization in a new event loop
        return asyncio.run(_async_main())
    
    except RuntimeError as error:
        if 'asyncio.run() cannot be called from a running event loop' in str(error):
            print(
                f'Warning: Could not initialize RoutingAgent with asyncio.run(): {error}. '
                'This can happen if an event loop is already running.'
            )
        raise

