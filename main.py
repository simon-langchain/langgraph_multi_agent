"""
Main entry point demonstrating correct checkpointer usage.

KEY POINTS:
1. Thread ID is passed in config during invocation, NOT in state
2. For InMemorySaver: You can omit checkpointer param (built-in) OR pass explicitly
3. For MySQL: You MUST pass checkpointer to compile()
"""
import uuid
from langchain_core.messages import HumanMessage
from agents.business_agent import create_business_agent_graph
from agents.database_agent import create_database_agent_graph
from utils.checkpointer import get_memory_saver, get_mysql_saver


def run_agent_with_memory(use_mysql: bool = False):
    """
    Demonstrate correct usage of checkpointer with conversational memory.

    Args:
        use_mysql: If True, use MySQL checkpointer. If False, use InMemorySaver.
    """
    # Create the workflow
    workflow = create_business_agent_graph()

    # CORRECT WAY TO COMPILE WITH CHECKPOINTER
    if use_mysql:
        # For MySQL: MUST pass checkpointer
        print("Using MySQL Checkpointer")
        checkpointer = get_mysql_saver("mysql://user:pass@localhost:3306/langgraph")
        graph = workflow.compile(checkpointer=checkpointer)
    else:
        # For InMemory: Two options

        # Option 1: Don't pass anything (uses built-in)
        # graph = workflow.compile()

        # Option 2: Pass explicitly (recommended for shared memory)
        print("Using InMemory Checkpointer")
        checkpointer = get_memory_saver()
        graph = workflow.compile(checkpointer=checkpointer)

    # Create a thread ID for this conversation
    # IMPORTANT: Thread ID goes in config, NOT in state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Starting conversation with thread_id: {thread_id} ===\n")

    # First query
    print("Query 1: What are the key supply chain metrics?")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What are the key supply chain metrics?")]},
        config=config  # Thread ID is here!
    )
    print(f"Response: {result1['messages'][-1].content}\n")

    # Follow-up query - this should maintain context
    print("Query 2: Can you explain the first one in more detail?")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Can you explain the first one in more detail?")]},
        config=config  # Same thread ID maintains context
    )
    print(f"Response: {result2['messages'][-1].content}\n")

    # Another follow-up to verify context is maintained
    print("Query 3: What about the others?")
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="What about the others?")]},
        config=config  # Same thread ID
    )
    print(f"Response: {result3['messages'][-1].content}\n")

    print("=== Conversation complete ===")
    print(f"All messages maintained context using thread_id: {thread_id}")

    return graph, thread_id


def demonstrate_new_conversation(graph, old_thread_id):
    """
    Demonstrate that a new thread ID starts a fresh conversation.
    """
    new_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": new_thread_id}}

    print(f"\n=== New conversation with thread_id: {new_thread_id} ===\n")
    print("Query: What about the others?")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What about the others?")]},
        config=config
    )
    print(f"Response: {result['messages'][-1].content}\n")
    print("Notice: The agent doesn't have context because it's a new thread_id")


if __name__ == "__main__":
    # Run with InMemory checkpointer
    graph, thread_id = run_agent_with_memory(use_mysql=False)

    # Demonstrate new conversation
    demonstrate_new_conversation(graph, thread_id)

    print("\n" + "="*60)
    print("To use MySQL checkpointer:")
    print("1. Set up MySQL database")
    print("2. Run: python main.py --mysql")
    print("3. Update connection string in utils/checkpointer.py")
    print("="*60)
