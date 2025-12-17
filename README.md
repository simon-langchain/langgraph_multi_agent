# LangGraph Multi-Agent System with Conversational Memory

This project demonstrates the **correct way** to build a LangGraph multi-agent system with conversational memory, addressing common issues with checkpointer configuration.

## 🎯 Key Issues Solved

Based on real-world implementation challenges:

1. **✓ Error: "LangGraph already has inbuilt Memory saver, it will be ignored"**
   - Solution: Don't pass `MemorySaver` to `compile()` (it's built-in) OR pass explicitly for shared memory

2. **✓ Context not retained between follow-up questions**
   - Solution: Use `add_messages` reducer in state and pass `thread_id` in config (not state)

3. **✓ Confusion about MySQL vs InMemory checkpointer usage**
   - Solution: Clear examples for both, with proper configuration

4. **✓ Uncertain about thread_id placement**
   - Solution: thread_id goes in `config`, not in `state`

## 📁 Project Structure

```
langgraph_multi_agent/
├── agents/
│   ├── business_agent/       # Handles business KB queries
│   │   ├── __init__.py
│   │   ├── state.py          # State with add_messages reducer
│   │   └── agent.py          # Agent graph definition
│   ├── database_agent/       # Handles structured data queries
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── agent.py
│   └── supervisor/           # Routes between agents
│       ├── __init__.py
│       ├── state.py
│       └── agent.py
├── api/
│   └── server.py             # FastAPI server for remote graph invocation
├── utils/
│   └── checkpointer.py       # Checkpointer configuration utilities
├── examples/
│   ├── mysql_example.py      # MySQL checkpointer examples
│   └── api_client_example.py # API client examples
├── main.py                   # Main entry point
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
cd langgraph_multi_agent

# Install dependencies
pip install langgraph langchain-openai fastapi uvicorn requests

# For MySQL support (optional)
pip install langgraph-checkpoint-mysql
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Basic Example

```bash
# Run with InMemory checkpointer
python main.py
```

### 4. Run the API Server

```bash
# Start server (run as module from project root)
python -m api.server

# Or with uvicorn directly
uvicorn api.server:app --host 0.0.0.0 --port 8000

# In another terminal, run the client examples
python examples/api_client_example.py
```

## 🔑 Critical Concepts

### 1. State Definition with add_messages Reducer

**❌ WRONG:**
```python
class State(TypedDict):
    messages: list[BaseMessage]  # Won't maintain conversation properly
```

**✓ CORRECT:**
```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ✓ Correct!
```

The `add_messages` reducer:
- Appends new messages to the list
- Handles message deduplication by ID
- Maintains conversation history properly

### 2. Checkpointer Configuration

#### Option A: InMemorySaver (Development/Testing)

**Method 1: Don't pass anything (uses built-in)**
```python
workflow = create_business_agent_graph()
graph = workflow.compile()  # ✓ Uses built-in MemorySaver
```

**Method 2: Pass explicitly (for shared memory)**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # ✓ Explicit shared memory
```

**⚠️ Warning:** If you pass `MemorySaver` unnecessarily, you'll see:
```
"LangGraph already has inbuilt Memory saver, it will be ignored"
```

#### Option B: MySQLSaver (Production)

**✓ CORRECT:**
```python
from langgraph.checkpoint.mysql import MySQLSaver

# Create checkpointer
checkpointer = MySQLSaver.from_conn_string(
    "mysql://user:password@host:port/database"
)

# MUST pass to compile()
graph = workflow.compile(checkpointer=checkpointer)
```

**❌ WRONG:**
```python
checkpointer = MySQLSaver.from_conn_string(conn_str)
graph = workflow.compile()  # Missing checkpointer! Won't use MySQL
```

### 3. Thread ID Placement

**❌ WRONG - thread_id in state:**
```python
result = graph.invoke({
    "messages": [HumanMessage(content="Hello")],
    "thread_id": "123"  # ❌ Won't work!
})
```

**✓ CORRECT - thread_id in config:**
```python
config = {"configurable": {"thread_id": "123"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config=config  # ✓ Correct!
)
```

### 4. Maintaining Conversation Context

```python
import uuid
from langchain_core.messages import HumanMessage

# Create thread for conversation
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# First query
result1 = graph.invoke(
    {"messages": [HumanMessage(content="What is supply chain?")]},
    config=config
)

# Follow-up query - SAME thread_id maintains context
result2 = graph.invoke(
    {"messages": [HumanMessage(content="Tell me more about the first point")]},
    config=config  # ✓ Agent has context from previous message
)

# New conversation - DIFFERENT thread_id
new_thread_id = str(uuid.uuid4())
new_config = {"configurable": {"thread_id": new_thread_id}}

result3 = graph.invoke(
    {"messages": [HumanMessage(content="Tell me more about the first point")]},
    config=new_config  # ✗ No context (different thread)
)
```

## 🌐 API Usage

### Start the Server

```bash
# Run as module from project root
python -m api.server

# Or with uvicorn
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Query an Agent

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are supply chain best practices?",
    "agent_type": "business"
  }'
```

Response:
```json
{
  "response": "Supply chain best practices include...",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_type": "business"
}
```

### Continue Conversation

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain the first one?",
    "thread_id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_type": "business"
  }'
```

### Get Conversation History

```bash
curl http://localhost:8000/conversation/550e8400-e29b-41d4-a716-446655440000?agent_type=business
```

## 🔧 MySQL Setup (Production)

### 1. Install MySQL

```bash
# macOS
brew install mysql
brew services start mysql

# Ubuntu/Debian
sudo apt-get install mysql-server
sudo systemctl start mysql
```

### 2. Create Database

```sql
mysql -u root -p

CREATE DATABASE langgraph_db;
CREATE USER 'langgraph_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON langgraph_db.* TO 'langgraph_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Update Connection String

In `utils/checkpointer.py`:
```python
connection_string = "mysql://langgraph_user:your_password@localhost:3306/langgraph_db"
```

### 4. Run with MySQL

```python
from utils.checkpointer import get_mysql_saver
from agents.business_agent import create_business_agent_graph

checkpointer = get_mysql_saver()
workflow = create_business_agent_graph()
graph = workflow.compile(checkpointer=checkpointer)
```

## 📚 Examples

### Example 1: Basic Conversation

```python
from main import run_agent_with_memory

# Run with InMemory checkpointer
graph, thread_id = run_agent_with_memory(use_mysql=False)
```

### Example 2: MySQL Checkpointer

```python
# See examples/mysql_example.py
python examples/mysql_example.py
```

### Example 3: API Client

```python
# See examples/api_client_example.py
python examples/api_client_example.py
```

## 🐛 Common Issues and Solutions

### Issue 1: "LangGraph already has inbuilt Memory saver"

**Cause:** Passing `MemorySaver` to `compile()` unnecessarily

**Solution:**
```python
# Option 1: Don't pass anything
graph = workflow.compile()

# Option 2: Pass for shared memory
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

### Issue 2: Context Not Maintained

**Cause:** Not using `add_messages` reducer or not passing `thread_id`

**Solution:**
```python
# 1. Use add_messages reducer in state
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Pass thread_id in config
config = {"configurable": {"thread_id": thread_id}}
graph.invoke(input, config=config)
```

### Issue 3: MySQL Connection Error

**Cause:** Incorrect connection string or database not created

**Solution:**
1. Verify MySQL is running: `mysql -u root -p`
2. Check database exists: `SHOW DATABASES;`
3. Verify connection string format:
   ```
   mysql://username:password@host:port/database
   ```

## 🏗️ Architecture Decisions

### Why Separate Agent Files?

- **Modularity**: Each agent is self-contained
- **Scalability**: Easy to add new agents
- **Testing**: Test agents independently
- **Deployment**: Deploy agents separately if needed

### Why Not Compile in Agent Files?

Agents return uncompiled workflows so the parent can:
- Choose appropriate checkpointer (Memory vs MySQL)
- Configure compilation options
- Compose multiple agents together

### Why FastAPI?

- **Standard**: Industry-standard Python web framework
- **Async**: Built-in async support for streaming
- **Docs**: Auto-generated API documentation
- **Remote**: Enables remote graph invocation

## 📖 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Checkpointer Guide](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MySQL Checkpointer](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-mysql)

## 🤝 Contributing

This example was built to address real-world implementation challenges. If you encounter issues or have suggestions, please open an issue.

## 📝 License

MIT License - feel free to use this as a template for your own projects.

---

**Built with ❤️ to help developers avoid common LangGraph pitfalls**
