from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
from datetime import datetime

# Import the tool calling agent
from tool_calling_agent import run_tool_calling_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tool Calling Agent API",
    description="API for running the tool calling agent.",
    version="1.0.0"
)

class AgentRequest(BaseModel):
    user_id: str
    conversation_id: str
    task: str
    max_iterations: Optional[int] = 10

@app.post("/invoke")
async def invoke_agent(request: AgentRequest) -> Dict[str, Any]:
    """
    Run the tool calling agent with the specified task, user_id, and conversation_id.
    """
    try:
        logger.info(f"🚀 Starting agent for user '{request.user_id}' with task: {request.task}")

        # The run_tool_calling_agent function will handle its own environment validation
        # and raise exceptions if required modules are missing.
        
        result = run_tool_calling_agent(
            task=request.task,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            max_iterations=10
        )

        logger.info(f"✅ Agent execution completed for user '{request.user_id}'.")
        return result

    except ImportError as ie:
        error_msg = f"A required library is not installed: {str(ie)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(
            status_code=501,
            detail=error_msg
        )
    except Exception as e:
        error_msg = f"An internal error occurred during agent execution: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 