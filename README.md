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
├── graphs/                   # LangGraph Studio entry point
│   ├── __init__.py
│   └── multi_agent_system.py # Unified graph with all agents
├── api/
│   └── server.py             # FastAPI server for remote graph invocation
├── utils/
│   └── checkpointer.py       # Checkpointer configuration utilities
├── examples/
│   ├── mysql_example.py      # MySQL checkpointer examples
│   └── api_client_example.py # API client examples
├── langgraph.json            # LangGraph Studio configuration
├── main.py                   # Main entry point
├── test_checkpointer.py      # Automated tests
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
cd langgraph_multi_agent

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install langgraph langchain-openai fastapi uvicorn requests langgraph-cli

# For MySQL support (optional)
pip install langgraph-checkpoint-mysql pymysql
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

### 5. Run with LangGraph Studio (Visual Development)

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Start LangGraph Studio
langgraph dev

# Open your browser to http://localhost:8123
# You'll see all three agents available:
# - business_agent
# - database_agent
# - supervisor
```

**LangGraph Studio Features**:
- 🎨 Visual graph representation
- 💬 Interactive chat interface with agents
- 🔍 Step-by-step execution debugging
- 📊 State inspection at each node
- 🔄 Thread management and conversation history
- ⏱️ Time-travel debugging (replay conversations)

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

## 🎨 LangGraph Studio

LangGraph Studio provides a visual interface for developing, debugging, and testing your agents. It's the easiest way to interact with your graphs.

### Starting LangGraph Studio

```bash
# Make sure you're in the project directory
cd langgraph_multi_agent

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Start LangGraph Studio
langgraph dev
```

The studio will start on `http://localhost:8123`

### The Multi-Agent System

The `langgraph.json` configuration exposes a unified **multi_agent_system** graph that shows all three agents working together:

- **supervisor** - Routes queries to the appropriate specialized agent
- **business_agent** - Handles business processes, policies, and KB queries
- **database_agent** - Handles structured data and SQL queries

This unified view lets you see the complete architecture and watch routing decisions in real-time!

### Using the Studio

**1. The Graph**
- You'll see the **multi_agent_system** with all three agents

**2. Visual Graph View**
- See the agent's node structure
- Understand the flow of execution
- Inspect state at each step

**3. Interactive Chat**
- Type queries in the chat interface
- Watch the agent process your request
- See intermediate steps and reasoning

**4. Thread Management**
- Create new conversation threads
- Switch between threads
- Each thread maintains separate context

**5. State Inspection**
- View the full state at any point
- See all messages in the conversation
- Inspect checkpointer data

**6. Time Travel Debugging**
- Replay conversations step-by-step
- Fork from any point in the conversation
- Test different paths

### Example Workflow

```bash
# 1. Start the studio
langgraph dev

# 2. Open http://localhost:8123 in your browser

# 3. Select "multi_agent_system" (shows all agents)

# 4. Try these queries to see routing in action:
#    - "What are the top supply chain metrics?" → Routes to business_agent
#    - "Show me sales data from last month" → Routes to database_agent
#    - "Explain the second metric in detail" → Uses context, stays with business

# 5. Watch the visual graph:
#    - See the supervisor node route to the appropriate agent
#    - Watch agents process and respond
#    - Observe routing decisions in real-time

# 6. Click on any node to see the state at that point

# 7. Use "New Thread" to start a fresh conversation
```

### Configuration

The `langgraph.json` file configures LangGraph Studio:

```json
{
  "dependencies": ["."],
  "graphs": {
    "multi_agent_system": "./graphs/multi_agent_system.py:graph"
  },
  "env": ".env"
}
```

- **dependencies**: Python packages to install
- **graphs**: Graph entry points (file:variable format)
- **env**: Environment file for API keys

**Important Note on Checkpointers**:
- ✅ **LangGraph Studio**: Graphs compile WITHOUT checkpointer (Studio provides persistence)
- ✅ **FastAPI/Code**: Graphs compile WITH checkpointer (you manage persistence)

The `graphs/` folder exports graphs without checkpointers for Studio use. For programmatic use (like `api/server.py`), compile with your chosen checkpointer.

### Troubleshooting

**Port already in use:**
```bash
langgraph dev --port 8124
```

**Can't find graphs:**
- Ensure you're in the project root directory
- Check that all graph files exist in `graphs/` folder
- Verify imports work: `python -c "from graphs.business_graph import graph"`

**API key errors:**
- Make sure OPENAI_API_KEY is set: `echo $OPENAI_API_KEY`
- Or create a `.env` file with `OPENAI_API_KEY=your-key`

## 🌐 API Usage

### Start the Server

```bash
# Run as module from project root
python -m api.server

# Or with uvicorn
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Two Routing Modes

The API server supports both automatic and manual routing:

#### 1. Automatic Routing (⭐ Recommended)

The supervisor analyzes the query and routes automatically (same as LangGraph Studio):

```bash
curl -X POST http://localhost:8000/query/auto \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are supply chain best practices?"
  }'
```

Response:
```json
{
  "response": "Supply chain best practices include...",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_type": "business_agent"
}
```

**Benefits:**
- ✅ Intelligent routing based on query content
- ✅ Same behavior as LangGraph Studio
- ✅ No need to specify agent type

#### 2. Manual Routing

Client specifies which agent to use:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are supply chain best practices?",
    "agent_type": "business"
  }'
```

**Use cases:**
- Direct control over routing
- Testing specific agents
- UI with explicit agent selection

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
