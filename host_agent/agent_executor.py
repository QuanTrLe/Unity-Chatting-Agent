import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from a2a.types import (
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
    InvalidParamsError,
)

from a2a.utils import( 
    new_agent_text_message,
    new_task
)
from a2a.utils.errors import ServerError

from routing_agent import get_initialized_routing_agent


logger = logging.getLogger(__name__)


class RoutingAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation"""
    
    def __init__(self):
        self.agent = get_initialized_routing_agent()
        
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue 
    ) -> None:
        # make sure that there will be no errors w push notifications
        error = self._validate_push_config(context)
        if error:
            raise ServerError(error = InvalidParamsError())
        
        # get components from context
        query = context.get_user_input()
        task = context.current_task
        
        # always produces a task, probably not practical but for testing
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        # invoking underlying agent and use streaming results as update events
        async for finished, text in self.agent.stream(query, task.context_id):
            # update task status as working if aint finished
            if not finished:
                await task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(text, task.context_id, task.id)
                )
                continue
                
            # else emit appropriate events
            await task_updater.add_artifact(
                [Part(root = TextPart(text = text))],
                name = 'response'
            )
            
            # mark task as complete and publishes final status update
            await task_updater.complete()
            break
    
    
    # Validation function that comprises of output check and push notification config check
    def _validate_push_config(self, context: RequestContext) -> bool:
        """True means invalid, false is valid."""
        
        push_notification_config = None
        if (context.configuration): 
            push_notification_config = context.configuration.push_notification_config
        
        if push_notification_config and not push_notification_config.url:
            logger.warning('Push notification URL is missing')
            return True
        
        return False
    
    
    # Example of supporting further features
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error = UnsupportedOperationError())