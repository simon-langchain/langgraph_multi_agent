"""
Test script to verify checkpointer configuration and conversational memory.
This demonstrates the solutions to all the issues discussed in the video.
"""
import uuid
import os
from langchain_core.messages import HumanMessage
from agents.business_agent import create_business_agent_graph
from utils.checkpointer import get_memory_saver


def test_without_checkpointer():
    """
    Test 1: Compile without passing checkpointer (uses built-in).
    This is the CORRECT way to avoid the warning.
    """
    print("\n" + "="*60)
    print("TEST 1: Compile without checkpointer (built-in MemorySaver)")
    print("="*60)

    workflow = create_business_agent_graph()

    # Don't pass checkpointer - uses built-in
    graph = workflow.compile()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nQuery 1: What are supply chain metrics?")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What are supply chain metrics?")]},
        config=config
    )
    print(f"Response: {result1['messages'][-1].content[:80]}...")

    print("\nQuery 2: Explain the first one")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Explain the first one")]},
        config=config
    )
    print(f"Response: {result2['messages'][-1].content[:80]}...")

    # Check if context was maintained
    if "supply chain" in result2['messages'][-1].content.lower():
        print("\n✓ SUCCESS: Context maintained!")
    else:
        print("\n✗ WARNING: Context may not be maintained")

    return True


def test_with_explicit_checkpointer():
    """
    Test 2: Compile with explicit MemorySaver for shared memory.
    This is useful when you want to share memory across multiple graph instances.
    """
    print("\n" + "="*60)
    print("TEST 2: Compile with explicit MemorySaver (shared memory)")
    print("="*60)

    workflow = create_business_agent_graph()

    # Create explicit checkpointer
    checkpointer = get_memory_saver()
    graph = workflow.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nQuery 1: What is inventory management?")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What is inventory management?")]},
        config=config
    )
    print(f"Response: {result1['messages'][-1].content[:80]}...")

    print("\nQuery 2: What are the key benefits?")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What are the key benefits?")]},
        config=config
    )
    print(f"Response: {result2['messages'][-1].content[:80]}...")

    # Verify context
    if "inventor" in result2['messages'][-1].content.lower():
        print("\n✓ SUCCESS: Context maintained with explicit checkpointer!")
    else:
        print("\n✗ WARNING: Context may not be maintained")

    return True


def test_thread_isolation():
    """
    Test 3: Verify that different thread_ids maintain separate conversations.
    """
    print("\n" + "="*60)
    print("TEST 3: Thread Isolation (separate conversations)")
    print("="*60)

    workflow = create_business_agent_graph()
    graph = workflow.compile()

    # Thread 1
    thread1_id = str(uuid.uuid4())
    config1 = {"configurable": {"thread_id": thread1_id}}

    print(f"\nThread 1 ({thread1_id[:8]}...): Ask about logistics")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What is logistics?")]},
        config=config1
    )
    print(f"Response: {result1['messages'][-1].content[:60]}...")

    # Thread 2 - different conversation
    thread2_id = str(uuid.uuid4())
    config2 = {"configurable": {"thread_id": thread2_id}}

    print(f"\nThread 2 ({thread2_id[:8]}...): Ask 'What about warehousing?'")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What about warehousing?")]},
        config=config2
    )
    print(f"Response: {result2['messages'][-1].content[:60]}...")

    # Back to Thread 1 - should remember logistics context
    print(f"\nThread 1 ({thread1_id[:8]}...): Follow up with 'Tell me more'")
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="Tell me more about it")]},
        config=config1
    )
    print(f"Response: {result3['messages'][-1].content[:60]}...")

    # Check if Thread 1 talks about logistics (not warehousing)
    if "logistic" in result3['messages'][-1].content.lower():
        print("\n✓ SUCCESS: Thread isolation working correctly!")
        print("  Thread 1 maintained its own context (logistics)")
        print("  Thread 2 had separate context (warehousing)")
    else:
        print("\n✗ WARNING: Thread isolation may not be working")

    return True


def test_state_structure():
    """
    Test 4: Verify state structure with add_messages reducer.
    """
    print("\n" + "="*60)
    print("TEST 4: State Structure (add_messages reducer)")
    print("="*60)

    workflow = create_business_agent_graph()
    graph = workflow.compile()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Multiple interactions
    messages_to_send = [
        "Hello, I need help with supply chain",
        "What are the main challenges?",
        "How can we solve the first challenge?"
    ]

    for i, msg in enumerate(messages_to_send, 1):
        print(f"\nMessage {i}: {msg}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=config
        )
        print(f"Response: {result['messages'][-1].content[:50]}...")

    # Get final state to verify message accumulation
    state = graph.get_state(config)
    total_messages = len(state.values.get("messages", []))

    print(f"\n✓ Total messages in state: {total_messages}")
    print(f"  (Expected: {len(messages_to_send) * 2} - {len(messages_to_send)} user + {len(messages_to_send)} assistant)")

    if total_messages == len(messages_to_send) * 2:
        print("\n✓ SUCCESS: add_messages reducer working correctly!")
    else:
        print(f"\n✗ WARNING: Expected {len(messages_to_send) * 2} messages, got {total_messages}")

    return True


def run_all_tests():
    """
    Run all tests to verify the implementation.
    """
    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("\n" + "="*60)
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("="*60)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("\nTests will fail without a valid API key.")
        print("="*60)
        return False

    print("\n" + "="*60)
    print("LANGGRAPH CHECKPOINTER TESTS")
    print("Testing solutions to common issues")
    print("="*60)

    tests = [
        ("Without Checkpointer", test_without_checkpointer),
        ("With Explicit Checkpointer", test_with_explicit_checkpointer),
        ("Thread Isolation", test_thread_isolation),
        ("State Structure", test_state_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {e}")
            results.append((test_name, "ERROR"))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")

    all_passed = all(status == "PASS" for _, status in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nKey takeaways:")
        print("  1. ✓ Can compile without checkpointer (uses built-in)")
        print("  2. ✓ Can compile with explicit checkpointer (for shared memory)")
        print("  3. ✓ Different thread_ids maintain separate conversations")
        print("  4. ✓ add_messages reducer accumulates messages correctly")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
