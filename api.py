from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import os
from datetime import datetime
import logging

# Import the advanced agent
from advanced_agent import run_enhanced_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph Advanced Agent API",
    description="API for running the enhanced LangGraph agent with phone calls and research",
    version="1.0.0"
)

class AgentRequest(BaseModel):
    task: str
    phone_number: Optional[str] = None
    max_steps: Optional[int] = 10
    call_id: Optional[str] = "none-provided"

class AgentResponse(BaseModel):
    success: bool
    task: str
    task_status: str
    completion_score: float
    steps_executed: int
    requirements: list
    key_facts_count: int
    research_results_count: int
    phone_calls_count: int
    execution_log: list
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LangGraph Advanced Agent API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with environment validation"""
    env_status = {
        "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "ELEVENLABS_API_KEY": bool(os.getenv("ELEVENLABS_API_KEY")),
        "ELEVENLABS_AGENT_ID": bool(os.getenv("ELEVENLABS_AGENT_ID")),
        "ELEVENLABS_PHONE_NUMBER_ID": bool(os.getenv("ELEVENLABS_PHONE_NUMBER_ID"))
    }
    
    all_configured = all(env_status.values())
    
    return {
        "status": "healthy" if all_configured else "partial",
        "environment": env_status,
        "ready": all_configured,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/run-agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run the advanced agent with the specified task
    
    Args:
        request: AgentRequest containing task details
        
    Returns:
        AgentResponse with execution results
    """
    try:
        logger.info(f"Starting agent execution for task: {request.task}")
        
        # Validate environment variables
        required_env_vars = [
            "MISTRAL_API_KEY",
            "TAVILY_API_KEY", 
            "ELEVENLABS_API_KEY",
            "ELEVENLABS_AGENT_ID",
            "ELEVENLABS_PHONE_NUMBER_ID"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise HTTPException(
                status_code=500,
                detail=f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        
        # Validate phone number - set to None if invalid
        validated_phone_number = request.phone_number
        if validated_phone_number and validated_phone_number in ["+1234567890", "1234567890"]:
            logger.info(f"Invalid phone number provided: {validated_phone_number}, setting to None")
            validated_phone_number = None
        
        # Run the agent in a separate thread to avoid blocking
        def run_agent_sync():
            return run_enhanced_agent(
                task=request.task,
                phone_number=validated_phone_number,
                max_steps=request.max_steps,
                call_id=request.call_id
            )
        
        # Execute the agent
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_agent_sync)
        
        logger.info(f"Agent execution completed with status: {result.get('task_status')}")
        
        # Format the response
        response = AgentResponse(
            success=True,
            task=result.get("current_task", request.task),
            task_status=result.get("task_status", "unknown"),
            completion_score=result.get("completion_score", 0.0),
            steps_executed=result.get("step_count", 0),
            requirements=[
                {
                    "name": req.get("name", ""),
                    "description": req.get("description", ""),
                    "completed": req.get("completed", False)
                }
                for req in result.get("task_requirements", [])
            ],
            key_facts_count=len(result.get("key_facts", {})),
            research_results_count=len(result.get("tavily_results", [])),
            phone_calls_count=len(result.get("phone_call_results", [])),
            execution_log=result.get("execution_log", [])
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )

@app.get("/examples")
async def get_examples():
    """Get example tasks that can be run with the agent"""
    return {
        "examples": [
            {
                "task": "order a pizza from dominos to 7 pearson square in W1 London",
                "description": "Order food delivery - requires research and phone call"
            },
            {
                "task": "call reception and let them know my mum is about to arrive, number 101 at 7 pearson square in W1 London",
                "description": "Notify reception - requires finding contact info and making call"
            },
            {
                "task": "book a table for 2 at a restaurant near Oxford Street London for tonight",
                "description": "Restaurant booking - requires research and reservation call"
            },
            {
                "task": "call a taxi to pick me up from Heathrow Airport Terminal 5",
                "description": "Transportation booking - requires finding taxi service and booking"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 