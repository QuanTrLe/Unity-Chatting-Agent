import json
from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse


PUBLIC_KEY = "your-very-secret-public-key"


class OAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Starlette,
        agent_card: AgentCard = None,
        public_paths: list[str] = None
    ):
        super().__init__(app)
        self.agent_card = agent_card
        self.public_paths = public_paths
        
        self.a2a_auth = {}
        # getting the security scopes
        if self.agent_card.security:
            for scope_dict in self.agent_card.security:
                self.a2a_auth = {'required_scopes': scope_dict.keys()}
    
    
    # function help respond when we get a request    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Allow public paths and anonymous access
        if path in self.public_paths or not self.a2a_auth:
            return await call_next(request)

        # Making sure that the request is in the right format / not missing
        auth_header = request.headers.get('Authorization') 
        print(f"Received auth header: {auth_header}")
        
        if not auth_header or not auth_header.startswith('OAuth '): # if the authen is missing
            return self._unauthorized(
                'Missing or malformed Authorization header.', request
            )
        
        # Checking the access token itself
        access_token = auth_header.split('OAuth ')[1]
        if not self._is_token_valid(access_token):
            return self._forbidden('Invalid or expired token.')
        
        return await call_next(request)


    def _is_token_valid(self, access_token: str) -> bool:
        # This is a placeholder for actual JWT validation.
        try:
            # A simple "validation": check if the token is a "JWT" with our public key in it
            payload = json.loads(access_token)
            return payload.get("key") == PUBLIC_KEY
        
        except (json.JSONDecodeError, AttributeError):
            return False
    
    
    # function handling forbidden 403 requests
    def _forbidden(self, reason: str, request: Request):
        accept_header = request.headers.get('accept', '')
        if 'text/event-stream' in accept_header:
            return PlainTextResponse(
                f'error forbidden: {reason}',
                status_code=403,
                media_type='text/event-stream',
            )
        return JSONResponse(
            {'error': 'forbidden', 'reason': reason}, status_code=403
        )
        
        
    # function handling unauthorized 401 requests
    def _unauthorized(self, reason: str, request: Request):
        accept_header = request.headers.get('accept', '')
        if 'text/event-stream' in accept_header:
            return PlainTextResponse(
                f'error unauthorized: {reason}',
                status_code=401,
                media_type='text/event-stream',
            )
        return JSONResponse(
            {'error': 'unauthorized', 'reason': reason}, status_code=401
        )
        