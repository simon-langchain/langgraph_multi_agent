"""
FastAPI server for remote graph invocation.

This demonstrates how to:
1. Deploy LangGraph agents as API endpoints
2. Maintain conversational memory across HTTP requests
3. Handle thread_id for conversation persistence
"""
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agents.business_agent import create_business_agent_graph
from agents.database_agent import create_database_agent_graph
from utils.checkpointer import get_memory_saver


# Initialize FastAPI app
app = FastAPI(title="LangGraph Multi-Agent API")

# Initialize agents with checkpointers
# IMPORTANT: Create checkpointer ONCE and reuse it across requests
checkpointer = get_memory_saver()
business_graph = create_business_agent_graph().compile(checkpointer=checkpointer)
database_graph = create_database_agent_graph().compile(checkpointer=checkpointer)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for agent queries."""
    message: str
    thread_id: Optional[str] = None
    agent_type: str = "business"  # "business" or "database"


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str
    thread_id: str
    agent_type: str


class ConversationHistory(BaseModel):
    """Model for conversation history."""
    messages: List[Dict[str, str]]
    thread_id: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "LangGraph Multi-Agent API",
        "available_agents": ["business", "database"]
    }


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query an agent with conversational memory.

    IMPORTANT:
    - If thread_id is provided, continues existing conversation
    - If thread_id is None, starts new conversation with generated thread_id
    - Thread ID is passed in config, not in state
    """
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())

        # Select the appropriate agent
        if request.agent_type == "business":
            graph = business_graph
        elif request.agent_type == "database":
            graph = database_graph
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_type: {request.agent_type}. Must be 'business' or 'database'"
            )

        # IMPORTANT: Thread ID goes in config
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        # Extract the response
        response_message = result["messages"][-1].content

        return QueryResponse(
            response=response_message,
            thread_id=thread_id,
            agent_type=request.agent_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """
    Stream responses from agent (for real-time updates).
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    if request.agent_type == "business":
        graph = business_graph
    elif request.agent_type == "database":
        graph = database_graph
    else:
        raise HTTPException(status_code=400, detail="Invalid agent_type")

    config = {"configurable": {"thread_id": thread_id}}

    # Stream the graph execution
    async def event_generator():
        for event in graph.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        ):
            yield f"data: {event}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/conversation/{thread_id}")
async def get_conversation_history(thread_id: str, agent_type: str = "business"):
    """
    Retrieve conversation history for a specific thread.
    """
    try:
        # Select agent
        if agent_type == "business":
            graph = business_graph
        elif agent_type == "database":
            graph = database_graph
        else:
            raise HTTPException(status_code=400, detail="Invalid agent_type")

        # Get state for this thread
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)

        if not state or not state.values.get("messages"):
            return {
                "messages": [],
                "thread_id": thread_id,
                "note": "No conversation history found"
            }

        # Format messages
        messages = []
        for msg in state.values["messages"]:
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })

        return ConversationHistory(
            messages=messages,
            thread_id=thread_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{thread_id}")
async def delete_conversation(thread_id: str):
    """
    Delete conversation history for a specific thread.
    """
    # Note: MemorySaver doesn't have a delete method
    # For production with MySQL, you would delete from database
    return {
        "message": f"Conversation {thread_id} deletion requested",
        "note": "With MemorySaver, conversations are cleared on restart. Use MySQL for persistent storage."
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting LangGraph Multi-Agent API Server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
