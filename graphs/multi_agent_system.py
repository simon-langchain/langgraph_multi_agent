"""
Multi-Agent System Graph for LangGraph Studio.

This graph composes the existing modular agents (business, database, supervisor)
into a unified system for visualization in LangGraph Studio.

IMPORTANT: This file imports and reuses the agent implementations from agents/
instead of duplicating their code, maintaining the modular structure.
"""
from typing import Literal
from langgraph.graph import StateGraph, START, END

# Import the existing agent components
from agents.business_agent import create_business_agent_graph, BusinessAgentState
from agents.database_agent import create_database_agent_graph, DatabaseAgentState
from agents.supervisor import create_supervisor_graph, SupervisorState


def route_supervisor(state: SupervisorState) -> Literal["business_agent", "database_agent", "__end__"]:
    """
    Route based on supervisor's decision.
    """
    next_choice = state.get("next", "").lower()

    if next_choice == "finish":
        return "__end__"
    elif next_choice == "business_agent":
        return "business_agent"
    elif next_choice == "database_agent":
        return "database_agent"
    else:
        return "__end__"


# Create the individual agent workflows (uncompiled)
supervisor_workflow = create_supervisor_graph()
business_workflow = create_business_agent_graph()
database_workflow = create_database_agent_graph()

# Compile each agent as a subgraph
supervisor_graph = supervisor_workflow.compile()
business_graph = business_workflow.compile()
database_graph = database_workflow.compile()

# Create the parent graph that orchestrates the agents
parent_workflow = StateGraph(SupervisorState)

# Add the compiled agent graphs as nodes
parent_workflow.add_node("supervisor", supervisor_graph)
parent_workflow.add_node("business_agent", business_graph)
parent_workflow.add_node("database_agent", database_graph)

# Set entry point to supervisor
parent_workflow.add_edge(START, "supervisor")

# Add conditional routing from supervisor to agents
parent_workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "business_agent": "business_agent",
        "database_agent": "database_agent",
        "__end__": END
    }
)

# After agents respond, they can either finish or continue
# For simplicity, we'll end after each agent response
parent_workflow.add_edge("business_agent", END)
parent_workflow.add_edge("database_agent", END)

# Compile the parent graph (Studio provides persistence)
graph = parent_workflow.compile()

# Export for LangGraph Studio
__all__ = ["graph"]
