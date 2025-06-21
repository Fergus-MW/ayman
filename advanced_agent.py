import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
import requests
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient
import re
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskRequirement:
    name: str
    description: str
    completed: bool = False
    result: Optional[Dict[str, Any]] = None


class AdvancedAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    current_task: str
    task_status: str
    task_requirements: List[Dict[str, Any]]
    completion_score: float
    key_facts: Dict[str, Any]
    tavily_results: List[Dict[str, Any]]
    phone_call_results: List[Dict[str, Any]]
    step_count: int
    max_steps: int
    execution_log: List[Dict[str, Any]]
    context_memory: Dict[str, Any]
    call_id: Optional[str]


# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize Mistral LLM
mistral_llm = ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
    temperature=0.3
)


@tool
def enhanced_tavily_search(query: str, max_results: int = 10, search_depth: str = "advanced") -> Dict[str, Any]:
    """
    Enhanced search using Tavily API with better error handling and result processing.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_depth: Search depth (basic, advanced)
        
    Returns:
        Dictionary containing processed search results
    """
    try:
        
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
            include_images=False
        )
        
        # Process and enhance results
        processed_results = []
        for result in response.get("results", []):
            processed_result = {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0),
                "relevance": "high" if result.get("score", 0) > 0.8 else "medium" if result.get("score", 0) > 0.5 else "low"
            }
            processed_results.append(processed_result)
        
        return {
            "success": True,
            "query": query,
            "answer": response.get("answer", ""),
            "results": processed_results,
            "result_count": len(processed_results),
            "timestamp": datetime.now().isoformat(),
            "search_depth": search_depth
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "timestamp": datetime.now().isoformat()
        }


@tool
def enhanced_elevenlabs_call(task: str, phone_number: str, 
                           system_prompt: str = "", first_message: str = "") -> Dict[str, Any]:
    """
    Enhanced outbound phone call function using ElevenLabs ConvAI API directly.
    
    Args:
        task: The task to be communicated during the call (stored as context)
        phone_number: The phone number to call (international format, e.g., 447874943523)
        system_prompt: The system prompt/instructions for the ElevenLabs agent
        first_message: The first message the agent will say when the call connects
        
    Returns:
        Enhanced call result with detailed information
    """
    try:
        
        # ElevenLabs ConvAI outbound call endpoint
        endpoint_url = "https://api.elevenlabs.io/v1/convai/twilio/outbound-call"
        
        # Prepare payload for ElevenLabs API
        payload = {
            "agent_id": os.getenv("ELEVENLABS_AGENT_ID"),
            "agent_phone_number_id": os.getenv("ELEVENLABS_PHONE_NUMBER_ID"),
            "to_number": phone_number
        }
        
        # Add optional parameters using conversation_config_override structure
        if system_prompt or first_message:
            payload["conversation_initiation_client_data"] = {
                "conversation_config_override": {
                    "agent": {}
                }
            }
            
            if system_prompt:
                payload["conversation_initiation_client_data"]["conversation_config_override"]["agent"]["prompt"] = {
                    "prompt": system_prompt
                }
            
            if first_message:
                payload["conversation_initiation_client_data"]["conversation_config_override"]["agent"]["first_message"] = first_message
        
        # Set required headers with user agent
        headers = {
            "Xi-Api-Key": os.getenv("ELEVENLABS_API_KEY"),
            "Api-Key": "xi-api-key",
            "Content-Type": "application/json",
            "User-Agent": "LangGraph-Agent/2.0"
        }
        
        # Make the API call
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "call_id": result.get("call_id"),
                "conversation_id": result.get("conversation_id"),
                "task": task,
                "phone_number": phone_number,
                "status": result.get("status", "initiated"),
                "message": "Enhanced ElevenLabs outbound call initiated successfully",
                "system_prompt": system_prompt,
                "first_message": first_message,
                "timestamp": datetime.now().isoformat(),
                "agent_id": os.getenv("ELEVENLABS_AGENT_ID"),
                "phone_number_id": os.getenv("ELEVENLABS_PHONE_NUMBER_ID"),
                "elevenlabs_response": result
            }
        else:
            return {
                "success": False,
                "error": f"ElevenLabs API Error {response.status_code}: {response.text}",
                "task": task,
                "phone_number": phone_number,
                "system_prompt": system_prompt,
                "first_message": first_message,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Exception calling ElevenLabs API: {str(e)}",
            "task": task,
            "phone_number": phone_number,
            "system_prompt": system_prompt,
            "first_message": first_message,
            "timestamp": datetime.now().isoformat()
        }


def analyze_task_requirements(task: str) -> List[TaskRequirement]:
    """
    Analyze the task and determine what requirements need to be fulfilled using Mistral.
    
    Args:
        task: The task description
        
    Returns:
        List of TaskRequirement objects
    """
    print(f"🔍 Analyzing task requirements for: {task}")
    
    # Use Mistral to intelligently analyze task requirements
    requirements_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert task analyzer. Analyze the given task and identify what specific requirements need to be fulfilled.

Common requirement types:
- phone_call: Making phone calls to people or businesses
- research: Gathering information or data online
- confirmation: Verifying or confirming information
- follow_up: Scheduling or arranging future actions
- email: Sending emails
- documentation: Creating or updating documents

For tasks like ordering food, booking services, or contacting businesses, you typically need:
1. Research to find contact information or details
2. Phone call to place the order or make the booking

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format. Do not include any other text:
{{
    "requirements": [
        {{
            "name": "requirement_type",
            "description": "What needs to be done"
        }}
    ]
}}"""),
        ("user", "Analyze this task and identify the requirements: {task}")
    ])
    
    try:
        chain = requirements_prompt | mistral_llm
        response = chain.invoke({"task": task})
        
        print(f"🤖 Mistral response: {response.content[:200]}...")
        
        # Parse the JSON response with robust extraction
        try:
            response_text = response.content.strip()
            
            # Try direct JSON parsing first
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                else:
                    raise Exception("No JSON found in response")
            
            requirements = []
            
            for req in result.get("requirements", []):
                requirements.append(TaskRequirement(
                    name=req.get("name", "unknown"),
                    description=req.get("description", "")
                ))
            
            print(f"✅ Mistral identified {len(requirements)} requirements")
            for i, req in enumerate(requirements, 1):
                print(f"   {i}. {req.name}: {req.description}")
            
            return requirements
            
        except Exception as json_error:
            print(f"⚠️ JSON parsing failed: {json_error}")
            print(f"Raw response: {response.content[:500]}...")
            raise Exception(f"Failed to parse Mistral response: {json_error}")
        
    except Exception as e:
        print(f"⚠️ Mistral analysis failed: {e}")
        print("🔄 Using fallback rule-based analysis...")
        
        # Enhanced fallback to rule-based analysis
        requirements = []
        task_lower = task.lower()
        
        # Pizza/food ordering requires research + phone call
        if any(keyword in task_lower for keyword in ["order", "pizza", "food", "delivery", "restaurant"]):
            requirements.append(TaskRequirement(
                name="research",
                description="Find restaurant contact information and menu details"
            ))
            requirements.append(TaskRequirement(
                name="phone_call",
                description="Call restaurant to place the order"
            ))
        
        # General phone call requirement
        elif any(keyword in task_lower for keyword in ["call", "phone", "contact", "speak"]):
            requirements.append(TaskRequirement(
                name="phone_call",
                description="Make a phone call to execute the task"
            ))
        
        # Research requirement
        if any(keyword in task_lower for keyword in ["research", "find", "search", "information", "details"]):
            if not any(req.name == "research" for req in requirements):
                requirements.append(TaskRequirement(
                    name="research",
                    description="Gather relevant information through research"
                ))
        
        # Booking/reservation tasks
        if any(keyword in task_lower for keyword in ["book", "reserve", "appointment", "schedule"]):
            if not any(req.name == "research" for req in requirements):
                requirements.append(TaskRequirement(
                    name="research",
                    description="Find booking information and availability"
                ))
            if not any(req.name == "phone_call" for req in requirements):
                requirements.append(TaskRequirement(
                    name="phone_call",
                    description="Call to make the booking or reservation"
                ))
        
        # Confirmation requirement
        if any(keyword in task_lower for keyword in ["confirm", "verify", "check", "validate"]):
            requirements.append(TaskRequirement(
                name="confirmation",
                description="Confirm or verify information"
            ))
        
        # Follow-up requirement
        if any(keyword in task_lower for keyword in ["follow up", "schedule", "arrange", "plan"]):
            requirements.append(TaskRequirement(
                name="follow_up",
                description="Arrange follow-up actions or scheduling"
            ))
        
        # Ensure we have at least one requirement
        if not requirements:
            requirements.append(TaskRequirement(
                name="phone_call",
                description="Execute the requested task via phone call"
            ))
        
        print(f"🔄 Fallback identified {len(requirements)} requirements:")
        for i, req in enumerate(requirements, 1):
            print(f"   {i}. {req.name}: {req.description}")
        
        return requirements


def calculate_completion_score(state: AdvancedAgentState) -> float:
    """
    Calculate a completion score based on fulfilled requirements and results quality using Mistral.
    
    Args:
        state: Current agent state
        
    Returns:
        Completion score between 0.0 and 1.0
    """
    requirements = state.get("task_requirements", [])
    task = state.get("current_task", "")
    
    if not requirements:
        return 0.0
    
    # Prepare context for Mistral scoring
    context = {
        "task": task,
        "total_requirements": len(requirements),
        "completed_requirements": sum(1 for req in requirements if req.get("completed", False)),
        "requirements_detail": json.dumps([
            {
                "name": req.get("name", ""),
                "description": req.get("description", ""),
                "completed": req.get("completed", False),
                "success": req.get("result", {}).get("success", False) if req.get("result") else False
            }
            for req in requirements
        ], indent=2)
    }
    
    scoring_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert task completion scorer. Analyze the task completion status and provide a score between 0.0 and 1.0.

Scoring criteria:
- 0.0: No progress made, no requirements started
- 0.2: Initial progress, some requirements started but not completed
- 0.4: Some requirements completed, but major ones remain
- 0.6: Most requirements completed, task is progressing well
- 0.8: Nearly complete, only minor items or verification remaining
- 1.0: Fully complete and successful, all requirements fulfilled

Consider both quantity (how many requirements completed) and quality (how successful the execution was).
For ordering/booking tasks, completion means the order/booking was successfully placed.

IMPORTANT: Respond with ONLY the numeric score as a decimal (e.g., 0.75). No other text."""),
        ("user", """Task: {task}

Requirements Status:
Total: {total_requirements}
Completed: {completed_requirements}

Details:
{requirements_detail}

Provide the completion score as a decimal number only.""")
    ])
    
    try:
        chain = scoring_prompt | mistral_llm
        response = chain.invoke(context)
        
        print(f"🎯 Mistral scoring response: {response.content}")
        
        # Parse the score with robust extraction
        score_text = response.content.strip()
        
        # Try to extract just the number
        import re
        number_match = re.search(r'(\d+\.?\d*)', score_text)
        if number_match:
            score = float(number_match.group(1))
        else:
            score = float(score_text)
        
        # Ensure score is within valid range
        final_score = max(0.0, min(1.0, score))
        print(f"📊 Calculated completion score: {final_score}")
        return final_score
        
    except Exception as e:
        # Fallback to rule-based scoring
        total_score = 0.0
        total_weight = 0.0
        
        for req in requirements:
            weight = 1.0
            req_score = 0.0
            
            if req.get("completed", False):
                req_score = 0.8  # Base score for completion
                
                # Bonus points for quality
                result = req.get("result", {})
                if req.get("name") == "phone_call" and result.get("success"):
                    req_score += 0.2
                elif req.get("name") == "research" and result.get("result_count", 0) > 0:
                    req_score += 0.2
                elif req.get("name") == "confirmation" and result.get("confirmed"):
                    req_score += 0.2
            
            total_score += req_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


def enhanced_planner_node(state: AdvancedAgentState):
    """
    Enhanced planning node with better task analysis and requirement tracking.
    """
    print(f"\n{'='*60}")
    print(f"🧠 PLANNER NODE - STEP {state.get('step_count', 0) + 1}")
    print(f"{'='*60}")
    
    messages = state.get("messages", [])
    current_task = state.get("current_task", "")
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)
    
    print(f"📋 Current Task: {current_task}")
    print(f"📊 Step: {step_count + 1}/{max_steps}")
    print(f"💬 Messages Count: {len(messages)}")
    print(f"🎯 Current Completion Score: {state.get('completion_score', 0.0):.2f}")
    
    # Extract task from messages if not set
    if not current_task and messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            current_task = last_message.content
    
    # Initialize task requirements if not set
    task_requirements = state.get("task_requirements", [])
    if not task_requirements:
        print("🔍 Analyzing task requirements...")
        requirements = analyze_task_requirements(current_task)
        task_requirements = [
            {
                "name": req.name,
                "description": req.description,
                "completed": req.completed,
                "result": req.result
            }
            for req in requirements
        ]
        print(f"📝 Identified {len(task_requirements)} requirements:")
        for i, req in enumerate(task_requirements, 1):
            print(f"   {i}. {req['name']}: {req['description']}")
    else:
        print(f"📝 Task Requirements ({len(task_requirements)}):")
        for i, req in enumerate(task_requirements, 1):
            status = "✅" if req['completed'] else "⏳"
            print(f"   {status} {i}. {req['name']}: {req['description']}")
    
    # Calculate completion score
    completion_score = calculate_completion_score(state)
    print(f"🎯 Updated Completion Score: {completion_score:.2f}")
    
    # Update execution log
    execution_log = state.get("execution_log", [])
    execution_log.append({
        "step": step_count + 1,
        "action": "planning",
        "completion_score": completion_score,
        "timestamp": datetime.now().isoformat()
    })
    
    # Determine task status
    task_status = TaskStatus.COMPLETED.value if completion_score >= 0.8 else TaskStatus.IN_PROGRESS.value
    print(f"📊 Task Status: {task_status}")
    
    # Show key facts and context
    key_facts = state.get("key_facts", {})
    context_memory = state.get("context_memory", {})
    print(f"🔑 Key Facts: {len(key_facts)} collected")
    print(f"🧠 Context Memory: {len(context_memory)} items")
    if key_facts:
        print("   Recent facts:")
        for k, v in list(key_facts.items()):
            print(f"     - {k}: {str(v)[:200]}{'...' if len(str(v)) > 100 else ''}")
    if context_memory:
        print("   Context items:")
        for k, v in list(context_memory.items()):
            print(f"     - {k}: {str(v)[:200]}{'...' if len(str(v)) > 100 else ''}")
    
    new_messages = []
    
    if task_status == TaskStatus.COMPLETED.value or step_count >= max_steps:
        status_msg = f"Task completed with score {completion_score:.2f}" if completion_score >= 0.8 else f"Task ended after {step_count} steps"
        new_messages.append(AIMessage(content=status_msg))
        
        return {
            "messages": new_messages,
            "current_task": current_task,
            "task_status": TaskStatus.COMPLETED.value,
            "task_requirements": task_requirements,
            "completion_score": completion_score,
            "step_count": step_count + 1,
            "execution_log": execution_log
        }
    else:
        incomplete_reqs = [req for req in task_requirements if not req["completed"]]
        plan_message = f"Step {step_count + 1}: Planning next actions. Incomplete requirements: {len(incomplete_reqs)}"
        new_messages.append(AIMessage(content=plan_message))
        
        return {
            "messages": new_messages,
            "current_task": current_task,
            "task_status": TaskStatus.IN_PROGRESS.value,
            "task_requirements": task_requirements,
            "completion_score": completion_score,
            "step_count": step_count + 1,
            "execution_log": execution_log
        }


def generate_intelligent_search_query(state: AdvancedAgentState) -> str:
    """
    Use Mistral AI to generate an intelligent search query based on all available context.
    
    Args:
        state: Current agent state with all context
        
    Returns:
        Optimized search query string
    """
    current_task = state.get("current_task", "")
    task_requirements = state.get("task_requirements", [])
    context_memory = state.get("context_memory", {})
    key_facts = state.get("key_facts", {})
    tavily_results = state.get("tavily_results", [])
    
    # Check if we need to find a phone number
    target_phone = os.getenv("TARGET_PHONE_NUMBER", "")
    need_phone_search = not target_phone or target_phone in ["+1234567890", "1234567890"]
    
    # Prepare context for Mistral
    context_info = {
        "task": current_task,
        "need_phone_search": need_phone_search,
        "previous_searches": [result.get("query", "") for result in tavily_results],
        "key_facts": {k: str(v)[:200] for k, v in key_facts.items()},
        "context_memory": {k: str(v)[:200] for k, v in context_memory.items()},
        "incomplete_requirements": [req.get("name", "") for req in task_requirements if not req.get("completed", False)],
        "search_count": len(tavily_results)
    }
    
    search_query_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert search query optimizer. Based on the task and all available context, generate the most effective search query for finding the needed information.

Key Guidelines:
1. If a phone number is needed, prioritize finding contact information
2. Include specific location details when available
3. Use business/service names when identifiable
4. Consider what information is still missing based on requirements
5. Build upon previous search results if available
6. Be specific and targeted rather than generic
7. Include relevant keywords that will improve search results

For phone number searches:
- Include business type, location, and specific services
- Use terms like "contact", "phone", "call", "booking" when appropriate

For information searches:
- Focus on the specific information needed
- Include context that will help find relevant results
- Use location and service-specific terms

IMPORTANT: Respond with ONLY the search query text. No explanations or extra text."""),
        ("user", """Generate an optimized search query for this context:

Task: {task}
Need Phone Search: {need_phone_search}
Previous Search Queries: {previous_searches}
Key Facts Available: {key_facts}
Context Memory: {context_memory}
Incomplete Requirements: {incomplete_requirements}
Search Attempt #: {search_count}

Generate the best search query for finding the missing information.""")
    ])
    
    try:
        print("🤖 Generating intelligent search query with Mistral AI...")
        
        chain = search_query_prompt | mistral_llm
        response = chain.invoke({
            "task": context_info["task"],
            "need_phone_search": context_info["need_phone_search"],
            "previous_searches": str(context_info["previous_searches"]),
            "key_facts": str(context_info["key_facts"]),
            "context_memory": str(context_info["context_memory"]),
            "incomplete_requirements": str(context_info["incomplete_requirements"]),
            "search_count": context_info["search_count"] + 1
        })
        
        search_query = response.content.strip()
        
        # Clean up the response (remove quotes, extra formatting)
        search_query = search_query.strip('"\'')
        
        # Ensure it's not too long
        if len(search_query) > 200:
            search_query = search_query[:200].rsplit(' ', 1)[0]
        
        # Ensure it's not empty
        if not search_query or len(search_query) < 5:
            raise Exception("Generated query too short or empty")
        
        print(f"✅ Mistral generated search query: {search_query}")
        return search_query
        
    except Exception as e:
        print(f"⚠️ Mistral query generation failed: {e}")
        print("🔄 Using fallback query generation...")
        
        # Enhanced fallback logic
        if need_phone_search:
            task_lower = current_task.lower()
            
            # Extract location information
            location_info = ""
            import re
            address_patterns = [
                r'to\s+([^,]+(?:,\s*[^,]+)*)',
                r'at\s+([^,]+(?:,\s*[^,]+)*)', 
                r'(\d+\s+[^,]+(?:,\s*[^,]+)*)'
            ]
            
            for pattern in address_patterns:
                match = re.search(pattern, current_task, re.IGNORECASE)
                if match:
                    location_info = match.group(1).strip()
                    break
            
            # Determine business type
            if "pizza" in task_lower:
                business_type = "pizza delivery restaurant"
            elif any(word in task_lower for word in ["reception", "hotel", "building", "apartment"]):
                business_type = "reception desk building management"
            elif any(word in task_lower for word in ["restaurant", "food", "order"]):
                business_type = "restaurant"
            elif any(word in task_lower for word in ["taxi", "cab", "uber"]):
                business_type = "taxi service"
            else:
                business_type = "business"
            
            if location_info:
                fallback_query = f"{business_type} phone number contact {location_info}"
            else:
                fallback_query = f"{business_type} phone number contact information London"
        else:
            # Regular information search
            fallback_query = current_task
            if context_memory.get("search_keywords"):
                fallback_query += f" {' '.join(context_memory['search_keywords'])}"
        
        print(f"🔄 Fallback query: {fallback_query}")
        return fallback_query


def enhanced_research_node(state: AdvancedAgentState):
    """
    Enhanced research node with AI-powered search query generation and better information processing.
    """
    print(f"\n{'='*60}")
    print(f"🔍 RESEARCH NODE")
    print(f"{'='*60}")
    
    current_task = state.get("current_task", "")
    task_requirements = state.get("task_requirements", [])
    context_memory = state.get("context_memory", {})
    key_facts = state.get("key_facts", {})
    
    print(f"📋 Task: {current_task}")
    print(f"🧠 Context Memory Items: {len(context_memory)}")
    print(f"🔑 Key Facts Available: {len(key_facts)}")
    print(f"📊 Previous Searches: {len(state.get('tavily_results', []))}")
    
    # Find research requirement
    research_req = next((req for req in task_requirements if req["name"] == "research"), None)
    if not research_req:
        return {"messages": [AIMessage(content="No research requirement found")]}
    
    # Show current context
    if context_memory:
        print("🧠 Available Context:")
        for k, v in list(context_memory.items())[:3]:
            print(f"   - {k}: {str(v)[:100]}...")
    
    if key_facts:
        print("🔑 Available Facts:")
        for k, v in list(key_facts.items())[:3]:
            print(f"   - {k}: {str(v)[:100]}...")
    
    # Check if we need to find a phone number (define here for use throughout function)
    target_phone = os.getenv("TARGET_PHONE_NUMBER", "")
    need_phone_search = not target_phone or target_phone in ["+1234567890", "1234567890"]
    
    # Generate intelligent search query using Mistral AI
    search_query = generate_intelligent_search_query(state)
    print(f"🎯 AI-Generated Search Query: {search_query}")
    print("🌐 Performing Tavily search...")
    
    # Store the query in context memory for future reference
    context_memory["last_generated_query"] = search_query
    context_memory["query_generation_method"] = "mistral_ai"
    context_memory["need_phone_search"] = need_phone_search
    
    # Perform enhanced search
    search_result = enhanced_tavily_search.invoke({"query": search_query, "max_results": 10, "search_depth": "advanced"})
    
    print(f"📊 Search Results: {search_result.get('success', False)}")
    if search_result.get("success"):
        print(f"   - Found {search_result.get('result_count', 0)} results")
        print(f"   - Answer: {search_result.get('answer', 'No answer')[:100]}...")
    else:
        print(f"   - Error: {search_result.get('error', 'Unknown error')}")
    
    # Update state
    tavily_results = state.get("tavily_results", [])
    tavily_results.append(search_result)
    
    # Extract key facts and look for phone numbers
    key_facts = state.get("key_facts", {})
    if search_result.get("success"):
        # Add search answer as a key fact
        if search_result.get("answer"):
            key_facts["search_answer"] = {
                "content": search_result["answer"],
                "query": search_query,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add top results as key facts and extract phone numbers
        phone_numbers_found = []
        for i, result in enumerate(search_result.get("results", [])[:3]):
            key_facts[f"research_result_{i+1}"] = {
                "title": result.get("title", ""),
                "content": result.get("content", "")[:300] + "...",
                "url": result.get("url", ""),
                "relevance": result.get("relevance", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Extract phone numbers from content
            content = result.get("content", "") + " " + result.get("title", "")
            import re
            # Look for various phone number patterns
            phone_patterns = [
                r'\+44\s*\d{2,4}\s*\d{3,4}\s*\d{3,4}',  # +44 format
                r'0\d{2,4}\s*\d{3,4}\s*\d{3,4}',        # UK format starting with 0
                r'\d{3,4}\s*\d{3,4}\s*\d{3,4}',         # General format
                r'\(\d{3,4}\)\s*\d{3,4}[-\s]*\d{3,4}'   # (xxx) xxx-xxxx format
            ]
            
            for pattern in phone_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    cleaned_phone = re.sub(r'[^\d+]', '', match)
                    if len(cleaned_phone) >= 10:  # Valid phone number length
                        phone_numbers_found.append({
                            "number": match.strip(),
                            "cleaned": cleaned_phone,
                            "source": result.get("title", ""),
                            "url": result.get("url", "")
                        })
        
        # Store found phone numbers
        if phone_numbers_found:
            key_facts["phone_numbers_found"] = phone_numbers_found
            print(f"📞 Found {len(phone_numbers_found)} phone numbers:")
            for phone in phone_numbers_found[:3]:  # Show first 3
                print(f"   - {phone['number']} from {phone['source']}")
            
            # Set the first found phone number as target if needed
            if need_phone_search and phone_numbers_found:
                best_phone = phone_numbers_found[0]
                # Convert to international format if needed
                phone_number = best_phone["cleaned"]
                if phone_number.startswith("0"):
                    phone_number = "+44" + phone_number[1:]
                elif not phone_number.startswith("+"):
                    phone_number = "+44" + phone_number
                
                os.environ["TARGET_PHONE_NUMBER"] = phone_number
                context_memory["found_phone_number"] = phone_number
                context_memory["phone_source"] = best_phone["source"]
                print(f"✅ Set target phone number: {phone_number}")
    
    # Update research requirement
    for req in task_requirements:
        if req["name"] == "research":
            req["completed"] = search_result.get("success", False)
            req["result"] = search_result
    
    # Update context memory
    context_memory["last_search_query"] = search_query
    context_memory["search_results_count"] = search_result.get("result_count", 0)
    context_memory["phone_search_performed"] = need_phone_search
    
    success_msg = f"Research completed for: {search_query}. Found {search_result.get('result_count', 0)} results."
    if phone_numbers_found:
        success_msg += f" Found {len(phone_numbers_found)} phone numbers."
    
    new_messages = [AIMessage(content=success_msg)]
    
    return {
        "messages": new_messages,
        "tavily_results": tavily_results,
        "key_facts": key_facts,
        "task_requirements": task_requirements,
        "context_memory": context_memory
    }


def enhanced_phone_call_node(state: AdvancedAgentState):
    """
    Enhanced phone call node with better call management and result tracking.
    """
    print(f"\n{'='*60}")
    print(f"📞 PHONE CALL NODE")
    print(f"{'='*60}")
    
    current_task = state.get("current_task", "")
    task_requirements = state.get("task_requirements", [])
    context_memory = state.get("context_memory", {})
    
    print(f"📋 Task: {current_task}")
    print(f"🧠 Context Memory: {len(context_memory)} items")
    print(f"🔑 Key Facts Available: {len(state.get('key_facts', {}))}")
    
    # Show available context
    if context_memory.get("search_answer"):
        print(f"📊 Research Context: {context_memory['search_answer']['content'][:150]}...")
    if state.get("key_facts"):
        print("🔑 Available Key Facts:")
        for k, v in list(state.get("key_facts", {}).items())[:3]:
            print(f"   - {k}: {str(v)[:100]}...")
    
    # Find phone call requirement
    phone_req = next((req for req in task_requirements if req["name"] == "phone_call"), None)
    if not phone_req:
        return {"messages": [AIMessage(content="No phone call requirement found")]}
    
    # Get phone number
    phone_number = os.getenv("TARGET_PHONE_NUMBER", "")
    
    # Check if we have a valid phone number
    if not phone_number or phone_number in ["+1234567890", "1234567890"]:
        # Try to get from context memory
        if context_memory.get("found_phone_number"):
            phone_number = context_memory["found_phone_number"]
            print(f"📱 Using found phone number: {phone_number}")
        else:
            # No phone number available
            print("❌ No phone number available - research may be needed first")
            new_messages = [AIMessage(content="Cannot make phone call: No phone number found. Research step may be needed first.")]
            return {"messages": new_messages}
    else:
        print(f"📱 Target Phone: {phone_number}")
    
    # Enhance task with context
    enhanced_task = current_task
    if context_memory.get("search_answer"):
        enhanced_task += f" Context: {context_memory['search_answer']}"
    
    print(f"📋 Enhanced Task: {enhanced_task}")
    
    # Generate task-specific content using Mistral
    print("\n🤖 Generating task-specific call content with Mistral AI...")
    try:
        task_content_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at generating task-specific content for phone calls. Based on the task and context, generate a task-specific section that will be inserted into a system prompt template.

The task-specific section should:
- Clearly explain what needs to be accomplished
- Include relevant details from the context
- Specify the desired outcome
- Be concise but complete (2-3 sentences max)

IMPORTANT: You MUST respond with ONLY the task-specific content text. Do not include any JSON formatting or extra text."""),
            ("user", """Task: {task}
Context: {context}

Generate the task-specific section for this call.""")
        ])
        
        context_info = ""
        if context_memory.get("search_answer"):
            context_info = f"Research findings: {context_memory['search_answer']['content']}"
        if key_facts := state.get("key_facts", {}):
            context_info += f" Key facts: {', '.join([f'{k}: {v}' for k, v in list(key_facts.items())[:3]])}"
        
        chain = task_content_prompt | mistral_llm
        response = chain.invoke({
            "task": enhanced_task,
            "context": context_info,
        })
        
        # Parse the response - no JSON needed, just get the text
        task_section = response.content.strip()
        
        # Validate the response
        if not task_section or len(task_section) < 10:
            raise Exception("Task section too short or empty")
        
        # Get call_id from state if available
        call_id = state.get("call_id")
        call_id_section = f"\n\nCALL ID: {call_id}" if call_id else ""
        
        # Build the complete system prompt using the template
        system_prompt = f"""You are a personal assistant called Ayman. You are making outbound calls on behalf of users. You are polite, efficient, and focused on completing tasks successfully.

CORE GUIDELINES:
- Always be assertive, direct and use colloquial language
- Listen actively and ask clarifying questions when needed
- Do not tell them you are making the call on behalf of a user, just focus on completing the task
- If the user provides insufficient information, infer or create the necessary details based on context and proceed. As long as the task is completed, it's up to you how this is achieved and you can decide the details yourself.

CURRENT TASK:
{task_section}

CONTEXT INFORMATION:
{context_info if context_info else 'No additional context available'}{call_id_section}

Remember to be helpful and professional throughout the entire conversation. Focus on completing the task while maintaining a positive interaction."""

        # Hardcoded first message
        first_message = "Hello!"

        print("✅ Mistral AI generated task-specific content:")
        print(f"\n📝 TASK SECTION ({len(task_section)} chars):")
        print("─" * 60)
        print(task_section)
        print("─" * 60)
        print(f"\n📝 COMPLETE SYSTEM PROMPT ({len(system_prompt)} chars):")
        print("─" * 60)
        print(system_prompt)
        print("─" * 60)
        print(f"\n💬 HARDCODED FIRST MESSAGE ({len(first_message)} chars):")
        print("─" * 60)
        print(first_message)
        print("─" * 60)
        
    except Exception as e:
        print(f"⚠️ Mistral AI failed: {str(e)}")
        print("🔄 Using fallback call configuration...")
        
        # Fallback task section
        task_section = f"I need to {enhanced_task.lower()}. Please help me complete this request efficiently and professionally."
        
        # Get call_id from state if available (for fallback)
        call_id = state.get("call_id")
        call_id_section = f"\n\nCALL ID: {call_id}" if call_id else ""
        
        # Build fallback system prompt using the same template
        system_prompt = f"""You are a personal assistant called Ayman. You are making outbound calls on behalf of users. You are polite, efficient, and focused on completing tasks successfully.

CORE GUIDELINES:
- Always be assertive, direct and use colloquial language
- Listen actively and ask clarifying questions when needed
- The person you are talking to is not the user, do not try to help them with their problems, just focus on completing the task
- Do not tell them you are making the call on behalf of a user, just focus on completing the task
- Do not say you are an assistant, AI, or automated system – speak naturally, like a real person. Be polite, friendly, and efficient. Do not overshare unless asked.
- If the user provides insufficient information, infer or create the necessary details based on context and proceed. As long as the task is completed, it's up to you how this is achieved and you can decide the details yourself.

CURRENT TASK:
{task_section}

CONTEXT INFORMATION:
{context_info if 'context_info' in locals() else 'No additional context available'}{call_id_section}

Remember to be helpful and professional throughout the entire conversation. Focus on completing the task while maintaining a positive interaction."""

        # Hardcoded first message
        first_message = "Hello!"
        
        print(f"\n📝 FALLBACK TASK SECTION ({len(task_section)} chars):")
        print("─" * 60)
        print(task_section)
        print("─" * 60)
        print(f"\n📝 FALLBACK SYSTEM PROMPT ({len(system_prompt)} chars):")
        print("─" * 60)
        print(system_prompt)
        print("─" * 60)
        print(f"\n💬 FALLBACK FIRST MESSAGE ({len(first_message)} chars):")
        print("─" * 60)
        print(first_message)
        print("─" * 60)
    
    print("\n📞 Making ElevenLabs API call...")
    print(f"🎯 Call Parameters:")
    print(f"   - Task: {enhanced_task[:100]}...")
    print(f"   - Phone: {phone_number}")
    print(f"   - System Prompt Length: {len(system_prompt)} chars")
    print(f"   - First Message Length: {len(first_message)} chars")
    
    # Make the call using ElevenLabs API
    call_result = enhanced_elevenlabs_call.invoke({
        "task": enhanced_task, 
        "phone_number": phone_number, 
        "system_prompt": system_prompt,
        "first_message": first_message
    })
    
    print(f"📞 Call Result: {call_result.get('success', False)}")
    if call_result.get("success"):
        print(f"   ✅ Call ID: {call_result.get('call_id', 'N/A')}")
        print(f"   📞 Conversation ID: {call_result.get('conversation_id', 'N/A')}")
        print(f"   📊 Status: {call_result.get('status', 'unknown')}")
    else:
        print(f"   ❌ Error: {call_result.get('error', 'Unknown error')}")
    
    # Update state
    phone_call_results = state.get("phone_call_results", [])
    phone_call_results.append(call_result)
    
    # Update key facts
    key_facts = state.get("key_facts", {})
    key_facts[f"phone_call_{len(phone_call_results)}"] = {
        "task": enhanced_task,
        "phone_number": phone_number,
        "success": call_result.get("success", False),
        "call_id": call_result.get("call_id", ""),
        "status": call_result.get("status", "unknown"),
        "timestamp": call_result.get("timestamp", "")
    }
    
    # Update phone call requirement
    for req in task_requirements:
        if req["name"] == "phone_call":
            req["completed"] = call_result.get("success", False)
            req["result"] = call_result
        # Also mark confirmation as completed if the call was successful
        elif req["name"] == "confirmation" and call_result.get("success", False):
            req["completed"] = True
            req["result"] = {"confirmed_via_call": True, "call_result": call_result}
    
    # Update context memory
    context_memory["last_call_id"] = call_result.get("call_id", "")
    context_memory["call_success"] = call_result.get("success", False)
    
    status = "successful" if call_result.get("success") else "failed"
    new_messages = [AIMessage(
        content=f"Phone call {status} to {phone_number}. Call ID: {call_result.get('call_id', 'N/A')}"
    )]
    
    return {
        "messages": new_messages,
        "phone_call_results": phone_call_results,
        "key_facts": key_facts,
        "task_requirements": task_requirements,
        "context_memory": context_memory
    }


def enhanced_should_continue(state: AdvancedAgentState) -> str:
    """
    Enhanced decision function with better logic for determining next steps.
    """
    print(f"\n{'='*60}")
    print(f"🤔 DECISION NODE - SHOULD CONTINUE?")
    print(f"{'='*60}")
    
    task_status = state.get("task_status", TaskStatus.PENDING.value)
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)
    completion_score = state.get("completion_score", 0.0)
    
    print(f"📊 Current Status: {task_status}")
    print(f"📈 Completion Score: {completion_score:.2f}")
    print(f"🔢 Step: {step_count}/{max_steps}")
    
    # Check completion conditions
    if task_status == TaskStatus.COMPLETED.value or completion_score >= 0.8 or step_count >= max_steps:
        print("🏁 ENDING: Task completed or max steps reached")
        return END
    
    # Check requirements
    task_requirements = state.get("task_requirements", [])
    incomplete_reqs = [req for req in task_requirements if not req["completed"]]
    
    print(f"📝 Requirements Status:")
    for req in task_requirements:
        status = "✅" if req["completed"] else "⏳"
        print(f"   {status} {req['name']}: {req['description']}")
    
    if not incomplete_reqs:
        print("🏁 ENDING: All requirements completed")
        return END
    
    # Prioritize actions based on dependencies
    req_names = [req["name"] for req in incomplete_reqs]
    print(f"⏳ Incomplete Requirements: {req_names}")
    
    # Research first if needed for other tasks
    if "research" in req_names and any(name in req_names for name in ["phone_call", "confirmation"]):
        print("🔍 NEXT: Research (needed for other tasks)")
        return "research"
    
    # Then phone calls
    if "phone_call" in req_names:
        print("📞 NEXT: Phone call")
        return "phone_call"
    
    # Then other requirements
    if "research" in req_names:
        print("🔍 NEXT: Research")
        return "research"
    
    # For confirmation or other requirements, try phone call
    if "confirmation" in req_names:
        print("📞 NEXT: Phone call for confirmation")
        return "phone_call"
    
    # If we have incomplete requirements but none match our nodes, end the task
    print("🏁 ENDING: No matching nodes for remaining requirements")
    return END


# Create enhanced workflow
enhanced_workflow = StateGraph(AdvancedAgentState)

# Add nodes
enhanced_workflow.add_node("planner", enhanced_planner_node)
enhanced_workflow.add_node("research", enhanced_research_node)
enhanced_workflow.add_node("phone_call", enhanced_phone_call_node)

# Add edges
enhanced_workflow.add_edge("research", "planner")
enhanced_workflow.add_edge("phone_call", "planner")

# Add conditional edges
enhanced_workflow.add_conditional_edges(
    "planner",
    enhanced_should_continue,
    {
        "research": "research",
        "phone_call": "phone_call",
        END: END
    }
)

# Set entry point
enhanced_workflow.set_entry_point("planner")

# Compile the enhanced graph
enhanced_app = enhanced_workflow.compile()


def run_enhanced_agent(task: str, phone_number: str = None, max_steps: int = 15, call_id: str = None) -> Dict[str, Any]:
    """
    Run the enhanced agent with a specific task.
    
    Args:
        task: The task to execute
        phone_number: Optional phone number to call
        max_steps: Maximum number of steps to execute
        call_id: Optional call ID to include in system prompt
        
    Returns:
        Final state of the agent with detailed results
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING ENHANCED LANGGRAPH AGENT")
    print(f"{'='*80}")
    print(f"📋 Task: {task}")
    print(f"📱 Phone Number: {phone_number or 'Will search for phone number'}")
    print(f"🔢 Max Steps: {max_steps}")
    print(f"🆔 Call ID: {call_id or 'Not provided'}")
    print(f"{'='*80}")
    
    if phone_number:
        os.environ["TARGET_PHONE_NUMBER"] = phone_number
    else:
        # Clear any existing phone number to force search
        os.environ.pop("TARGET_PHONE_NUMBER", None)
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_task": task,
        "task_status": TaskStatus.PENDING.value,
        "task_requirements": [],
        "completion_score": 0.0,
        "key_facts": {},
        "tavily_results": [],
        "phone_call_results": [],
        "step_count": 0,
        "max_steps": max_steps,
        "execution_log": [],
        "context_memory": {},
        "call_id": call_id
    }
    
    print("🔄 Initial State Created")
    print("🚀 Running enhanced workflow...")
    
    # Run the enhanced workflow
    final_state = enhanced_app.invoke(initial_state)
    
    print(f"\n{'='*80}")
    print(f"🏁 AGENT EXECUTION COMPLETED")
    print(f"{'='*80}")
    
    return final_state


if __name__ == "__main__":
    # Example usage
    task = "order a medium margarita pizza from Pizza Hut to 17 Rock St, Kemptown, Brighton BN2 1NF"
    phone_number = None  # Let the agent find the phone number via research
    
    print("Starting enhanced LangGraph agent...")
    result = run_enhanced_agent(task, phone_number, max_steps=10)
    
    print("\n" + "="*50)
    print("AGENT EXECUTION COMPLETED")
    print("="*50)
    print(f"Task: {result['current_task']}")
    print(f"Status: {result['task_status']}")
    print(f"Completion Score: {result['completion_score']:.2f}")
    print(f"Steps Executed: {result['step_count']}")
    
    print("\nTask Requirements:")
    for req in result.get('task_requirements', []):
        status = "✓" if req['completed'] else "✗"
        print(f"  {status} {req['name']}: {req['description']}")
    
    print(f"\nKey Facts Collected: {len(result.get('key_facts', {}))}")
    print(f"Research Results: {len(result.get('tavily_results', []))}")
    print(f"Phone Calls Made: {len(result.get('phone_call_results', []))}")
    
    print("\nExecution Log:")
    for log_entry in result.get('execution_log', []):
        print(f"  Step {log_entry['step']}: {log_entry['action']} (Score: {log_entry.get('completion_score', 0):.2f})")
    
    print("\n" + "="*50) 