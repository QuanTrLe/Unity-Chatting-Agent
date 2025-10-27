import logging
import os
from dotenv import load_dotenv

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


load_dotenv()
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

memory = MemorySaver()


# For us to parse out the state of the task and give the correct response back to supervisor 
class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class WeatherAgent:
    """Weather Agent"""
    
    # beginning prompt
    SYSTEM_INSTRUCTION = """
    You are a specialized Weathers agent. Your primary function is to utilize the provided tools to 
    search for Weathers and their details while answering related questions.
    
    You must rely exclusively on these tools for information, do not invent forecasts or weather information.
    You can however, answer basic questions based on the information that you retrieved from the tool.
    
    Always try to avoid having to ask for user clarifications before moving onto the next steps.
    
    You can also search for simple question related to the topic by using the search tool if needed
    when asked for clarification by the tools. Do this before asking the user for input, 
    especially when you need the coordinates of a location / city, or a state's two letter code. 
    However do not simply invent information in this step, nor assume any unknown information.
    Examples:
        1) 
        Query: "What's the weather at Lehigh University, Bethlehem, PA in USA". 
        Work Process: "Search online for coordinates of Lehigh University or Bethlehem. 
            Then use tool to get weather at coordinates." 
    
    Try to break the user's query into smaller steps that can be used for the tools.
    Especially try to do this if the tools does not support what you are trying to do.
    When asked to get the weather for a trip, take the forecast at the destination
    Examples:
        1)
        Query: "Will Bethlehem, PA in USA hotter than Philadelphia next week"
        Broken into: "Get the forecast for Bethlehem, PA in USA. Get forecast for Philadelphia.
            Then compare the average temperature between the two cities."
        2)
        Query: "I'm travelling from Bethlehem, PA to Philadelphia. Get me a day that's the coolest but is still sunny"
        Broken into: "Get the forecast for Philadelphia. Out of the forecast, choose days that are sunny.
            Then choose the sunny day that has the lowest temperature."
    
    Ensure that your Markdown-formatted response includes all relevant tool output.
    """
    
    # prompt for the agent to follow the response format
    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as "completed" if the request is fully addressed and no further input is needed. '
        'Select status as "input_required" if you need more information from the user or are asking a clarifying question. '
        'Select status as "error" if an error occurred or the request cannot be fulfilled.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
    
    
    # Initialize the model and assign the mcp tools
    def __init__(self, mcp_tools: list[Any]):
        """Initializes the Weather Agent 
        Args:
            mcp_tools: A list of preloaded mcp tools provided by main.
        """
        
        logger.info('Initializing Weather Agent with preloaded MCP tools...')
        
        # check for google API Key
        try:
            model = os.getenv('GOOGLE_GENAI_MODEL')
            if not model:
                raise ValueError('GOOGLE_GENAI_MODEL environment variable is not set')
            
            if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
                # If not using Vertex AI, initialize with Google Generative AI
                logger.info('ChatVertexAI model initialized successfully.')
                self.model = ChatVertexAI(model=model)
            
            else:
                # Using the model name from your provided file
                self.model = ChatGoogleGenerativeAI(model=model)
                logger.info(
                    'ChatGoogleGenerativeAI model initialized successfully.'
                )
        
        # error case
        except Exception as error:
            logger.error(
                f'Failed to initialize ChatGoogleGenerativeAI model: {error}',
                exc_info=True,
            )
            raise

        # making sure we did get mcp tools
        self.mcp_tools = mcp_tools
        if not self.mcp_tools:
            raise ValueError('No MCP tools provided to Weather Agent')
    
    
    # Creating the react agent, feed it the user query and get the response back
    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        logger.info(f"Weather.ainvoke called with query: '{query}', session_id: '{session_id}'")
        
        # create and ainvoke the agent
        try:
            # create the agent itself
            airbnb_agent_runnable = create_react_agent(
                model = self.model,
                tools = self.mcp_tools,
                checkpointer = memory,
                prompt = self.SYSTEM_INSTRUCTION,
                response_format = ( # in format tuple[str, StructuredResponseSchema]
                    self.RESPONSE_FORMAT_INSTRUCTION,
                    ResponseFormat
                )
            )
            logger.debug('LangGraph React agent for Weather task created with preloaded tools.')
            
            # starting ainvoke
            config : RunnableConfig = {'configurable': {'thread_id': session_id}} # type casting config
            langgraph_input = {'messages': [('user', query)]} # input for graph to run
            logger.debug(
                f'Invoking Weather agent with input: {langgraph_input} and config: {config}'
            )
            
            await airbnb_agent_runnable.ainvoke(input = langgraph_input, config = config)
            logger.debug(
                'Weather Agent ainvoke call completed. Fetching response from state...'
            )
            
            # parse the runnable result to get agent response
            response = self._get_agent_response_from_state(
                config, airbnb_agent_runnable
            )
            logger.info(
                f'Response from Weather Agent state for session {session_id}: {response}'
            )
            return response
        
        # for error, mark as complete with error message
        except httpx.HTTPStatusError as http_err:
            logger.error(
                f'HTTPStatusError in Weather agent ainvoke (Weather task): {http_err.response.status_code} - {http_err}',
                exc_info=True,
            )
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'An error occurred with an external service for Weather task: {http_err.response.status_code}',
            }
        except Exception as error:
            logger.error(
                f'Unhandled exception in Weather agent ainvoke (Weather task): {type(error).__name__} - {error}',
                exc_info=True,
            )
            return {
                'is_task_complete': True,  # or could mark as False, errored
                'require_user_input': False,
                'content': f'An unexpected error occurred while processing your Weather task: {type(error).__name__}.',
            }
    
    
    # from the state given parse out the response according to the response format we defined
    # if it doesnt follow the format then use the AIMessage streamed instead
    def _get_agent_response_from_state(self, config: RunnableConfig, agent_runnable):
        """Retrieves and formats the agent's response from the state of the given agent_runnable."""
        
        logger.debug(
            f'Entering _get_agent_response_from_state for config: {config} using agent: {type(agent_runnable).__name__}'
        )

        try:
            # make sure runnable agent have get state, else mark as error
            if not hasattr(agent_runnable, 'get_state'):
                logger.error(
                    f'Agent runnable of type {type(agent_runnable).__name__} does not have get_state method.'
                )
                return { # terminal
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': 'Internal error: Agent state retrieval misconfigured.',
                }
            
            # get state snapshot and values from it
            current_state_snapshot = agent_runnable.get_state(config)
            state_values = getattr(current_state_snapshot, 'values', None)
            logger.debug(
                f'Retrieved state snapshot values: {"Available" if state_values else "Not available or None"}'
            )
        
        # in error case markign as complete terminally
        except Exception as error:
            logger.error(
                f'Error getting state from agent_runnable ({type(agent_runnable).__name__}): {error}',
                exc_info=True,
            )
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': 'Error: Could not retrieve agent state.',
            }
        
        #make sure that the state values arent empty, which would be an error
        if not state_values:
            logger.error(
                f'No state values found for config: {config} from agent {type(agent_runnable).__name__}'
            )
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': 'Error: Agent state is unavailable.',
            }
        
        # getting the structured response out of the state values now that we've got it
        structured_response = None
        if isinstance(state_values, dict): 
            structured_response = state_values.get('structured_response')
        else: 
            structured_response = getattr(state_values, 'structured_response', None)
        
        # get the data now that we've got a structured response format
        if structured_response and isinstance(structured_response, ResponseFormat):
            logger.info(
                f'Formatted response from structured_response: {structured_response}'
            )
            
            # if task is completed
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message
                }
            
            # if task requires input or is in error
            return {
                'is_task_complete': False,
                'require_user_input': structured_response.status == 'input_required',
                'content': structured_response.message # will be error message if status is error
            }
        
        # fallback case when structured response is in unexpected format or None
        # in this case we get and response with the ai message instead
        final_messages = []
        if isinstance(state_values, dict): final_messages = state_values.get('messages', [])
        else: final_messages = getattr(state_values, 'messages', [])
        
        if final_messages and isinstance(final_messages[-1], AIMessage):
            ai_content = final_messages[-1].content
            # case of just a simple str message
            if ai_content and isinstance(ai_content, str):
                logger.warning(
                    f'Structured response not found or not in ResponseFormat. Falling back to last AI message content for config {config}.'
                )
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': ai_content,
                }
            
            # case of list of tool calls
            if ai_content and isinstance(ai_content, list):
                text_parts = []
                for part in ai_content:
                    if isinstance(part, dict) and part.get('type') == 'text': 
                        text_parts.append(part['text'])
                
                if text_parts:
                    logger.warning(
                        f'Structured response not found. Falling back to concatenated text from last AI message parts for config {config}.'
                    )
                    return {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': '\n'.join(text_parts),
                    }
            
        # Fallback case, shouldn't happen so mark as require user input for error
        logger.warning(
            f'Structured response not found or not in expected format, and no suitable fallback AI message. State for config {config}: {state_values}'
        )
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment due to an unexpected response format. Please try again.'
        }
        
        
    # Create react agent, feed query and get response from stream states
    async def stream(self, query: str, session_id: str) -> AsyncIterable[Any]:
        logger.info(
            f"WeatherAgent.stream called with query: '{query}', sessionId: '{session_id}'"
        )
        
        # initializing the agent and configurations we need
        agent_runnable = create_react_agent(
            model = self.model,
            tools = self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(
                self.RESPONSE_FORMAT_INSTRUCTION,
                ResponseFormat,
            )
        )
        config: RunnableConfig = {'configurable': {"thread_id": session_id}}
        langgraph_input = {'messages': [('user', query)]}
        
        # streaming and getting response
        logger.debug(
            f'Streaming from Weather Agent with input: {langgraph_input} and config: {config}'
        )
        try:
            async for chunk in agent_runnable.astream_events(
                langgraph_input, config, version = 'v1'
            ):
                logger.debug(f'Stream chunk for {session_id}: {chunk}')
                
                # basing off the event names that were streamed since they let us know event type
                event_name = chunk.get('event')
                data = chunk.get('data', {}) # the streamed chunk
                content_to_yield = None # content to return back
                
                # if we're using a tool
                if event_name == 'on_tool_start':
                    tool_name = data.get('name', 'a tool')
                    content_to_yield = f'Using tool: {tool_name}...'
                
                # if a simple stream message
                elif event_name == 'on_chat_model_stream':
                    message_chunk = data.get('chunk')
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        content_to_yield = message_chunk.content
                
                # if we did get something to return from the stream
                if content_to_yield:
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': content_to_yield
                    }
            
            # After events get the final structured response from the agent's state
            final_response = self._get_agent_response_from_state(config, agent_runnable)
            logger.info(
                f'Final response from state after stream for session {session_id}: {final_response}'
            )
            yield final_response 
            
        except Exception as error:
            logger.error(
                f'Error during WeatherAgent.stream for session {session_id}: {error}',
                exc_info=True,
            )
            yield {
                'is_task_complete': True,  # Stream ended due to error
                'require_user_input': False,
                'content': f'An error occurred during streaming: {getattr(error, "message", str(error))}',
            }