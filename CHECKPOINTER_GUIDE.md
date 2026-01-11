# Checkpointer Usage Guide: Studio vs Production

## Overview

The **#1 cause of memory issues** in LangGraph multi-agent systems is incorrect checkpointer usage based on deployment context.

This guide clarifies when to use checkpointers and when not to.

---

## Quick Reference

| Context | Checkpointer? | Example File |
|---------|--------------|--------------|
| **LangGraph Studio** (`langgraph dev`) | ❌ NO | `graphs/multi_agent_system.py` |
| **Direct Python** (`python main.py`) | ✅ YES | `main.py` |
| **Production API** (FastAPI) | ✅ YES | `api/server.py` |

---

## Context 1: LangGraph Studio

**Command**: `langgraph dev`

**Where**: `graphs/multi_agent_system.py`

**Checkpointer**: ❌ **NO** - Studio provides built-in persistence

### ✅ CORRECT
```python
# graphs/multi_agent_system.py

workflow = create_business_agent_graph()

# DO NOT pass checkpointer
graph = workflow.compile()  # Studio handles persistence

__all__ = ["graph"]
```

### ❌ WRONG
```python
# graphs/multi_agent_system.py

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # ❌ Conflicts!

# Error: "LangGraph already has inbuilt Memory saver"
```

### Why?

LangGraph Studio (`langgraph dev`) has its own built-in persistence layer. When you pass a checkpointer, it creates a conflict because the system tries to use two persistence mechanisms simultaneously.

---

## Context 2: Direct Python Execution

**Command**: `python main.py`

**Where**: `main.py` or your script

**Checkpointer**: ✅ **YES** - You manage persistence

### ✅ CORRECT
```python
# main.py

from langgraph.checkpoint.memory import MemorySaver

workflow = create_business_agent_graph()

# DO pass checkpointer
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

### ❌ WRONG
```python
# main.py

workflow = create_business_agent_graph()

# No checkpointer = no persistence!
graph = workflow.compile()  # ❌ Conversations not saved
```

### Why?

When running directly with Python (not via Studio), there's no built-in persistence. You must explicitly provide a checkpointer for conversational memory to work.

---

## Context 3: Production Deployment

**Command**: `uvicorn api.server:app` or production server

**Where**: `api/server.py`

**Checkpointer**: ✅ **YES** - Use MySQL for persistence

---

> **Note**: `langgraph-checkpoint-mysql` is a community-maintained package
> (https://github.com/tjni/langgraph-checkpoint-mysql), not an official
> LangChain library. For officially supported production persistence,
> consider using `PostgresSaver` from `langgraph-checkpoint-postgres`.

---

### ✅ CORRECT
```python
# api/server.py

from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

workflow = create_business_agent_graph()

# DO pass checkpointer for production
checkpointer = PyMySQLSaver.from_conn_string(
    "mysql://user:password@host:port/database"
)
graph = workflow.compile(checkpointer=checkpointer)
```

### ✅ ALSO CORRECT (for testing production setup)
```python
# api/server.py

from langgraph.checkpoint.memory import MemorySaver

# Use MemorySaver for testing production API
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

### Why?

Production needs persistent storage that survives server restarts. MySQL provides this. PyMySQLSaver works for testing the production setup but won't persist across restarts.

---

## Common Mistakes

### Mistake 1: Using Checkpointer with Studio

**Symptom**: Error message like "LangGraph already has inbuilt Memory saver"

**Cause**:
```python
# graphs/multi_agent_system.py (loaded by langgraph dev)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # ❌
```

**Fix**:
```python
# graphs/multi_agent_system.py
graph = workflow.compile()  # ✅ No checkpointer for Studio
```

---

### Mistake 2: Not Using Checkpointer in Production

**Symptom**: Conversations lost on server restart, no memory

**Cause**:
```python
# api/server.py
graph = workflow.compile()  # ❌ No persistence
```

**Fix**:
```python
# api/server.py
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
checkpointer = PyMySQLSaver.from_conn_string(...)
graph = workflow.compile(checkpointer=checkpointer)  # ✅
```

---

### Mistake 3: Using Same Code for Studio and Production

**Problem**: One codebase, different deployment contexts

**Solution**: Separate graph definition from compilation

```python
# graphs/multi_agent_system.py (for Studio)
def create_multi_agent_graph():
    """Return uncompiled workflow"""
    workflow = create_business_agent_graph()
    return workflow

# Export compiled version (no checkpointer for Studio)
graph = create_multi_agent_graph().compile()
__all__ = ["graph"]
```

```python
# api/server.py (for Production)
from graphs.multi_agent_system import create_multi_agent_graph
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

# Compile with checkpointer for production
workflow = create_multi_agent_graph()
checkpointer = PyMySQLSaver.from_conn_string(...)
graph = workflow.compile(checkpointer=checkpointer)
```

---

## File Organization

```
your_project/
├── graphs/
│   └── multi_agent_system.py   # ❌ NO checkpointer (for Studio)
├── main.py                      # ✅ YES checkpointer (for direct Python)
└── api/
    └── server.py                # ✅ YES checkpointer (for production)
```

### graphs/multi_agent_system.py
- **Purpose**: LangGraph Studio entry point
- **Checkpointer**: ❌ NO
- **Why**: Studio provides persistence

### main.py
- **Purpose**: Direct Python execution
- **Checkpointer**: ✅ YES (MemorySaver)
- **Why**: You manage persistence

### api/server.py
- **Purpose**: Production API
- **Checkpointer**: ✅ YES (PyMySQLSaver)
- **Why**: Production needs persistent storage

---

## Testing Your Setup

### Test Studio (No Checkpointer)
```bash
# Should work without errors
langgraph dev

# Open http://localhost:8123
# Send message: "Hello"
# Send follow-up: "What did I say?"
# Should maintain context ✅
```

### Test Direct Python (With Checkpointer)
```bash
# Should work without errors
python main.py

# Should see:
# "Using InMemory Checkpointer"
# Conversation with context maintained ✅
```

### Test Production API (With Checkpointer)
```bash
# Start server
uvicorn api.server:app --reload

# Test endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "agent_type": "business"}'

# Should return response with thread_id ✅
```

---

## Debugging

### Error: "LangGraph already has inbuilt Memory saver, it will be ignored"

**Full Error Message:**
```
ValueError: Heads up! Your graph '{name}' from '{path}' includes a custom 
checkpointer (type <class '...'). With LangGraph API, persistence is handled 
automatically by the platform, so providing a custom checkpointer here isn't 
necessary and will be ignored when deployed.
```

**What this means:**
You're running `langgraph dev` (LangGraph Studio) AND passing an explicit checkpointer in your graph definition. This creates a conflict because Studio provides its own persistence layer.

**Cause**: Using checkpointer with Studio

**Why it happens:**
- Your graph file (e.g., `graphs/multi_agent_system.py`) includes `checkpointer=MemorySaver()` or similar
- You're running with `langgraph dev` command
- Studio tries to use its built-in persistence, but you've also provided a checkpointer
- The conflict causes a ValueError that blocks execution

**Fix**: Remove checkpointer from graph file loaded by `langgraph dev`

```python
# In graphs/multi_agent_system.py (or wherever Studio loads your graph)

# ❌ REMOVE THIS:
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # Don't do this!

# ✅ USE THIS:
graph = workflow.compile()  # No checkpointer - Studio handles it
```

---

### "Context not maintained" / Follow-up questions don't work

**Symptoms:**
- First question works fine
- Follow-up questions don't maintain context
- Agent doesn't remember previous conversation
- Each query is treated as a new conversation

**Cause**: No checkpointer in direct Python/production execution

**Common scenarios:**
1. Running `python main.py` without passing checkpointer
2. FastAPI server not configured with checkpointer
3. Forgot to pass `config` with `thread_id` during invocation

**Fix 1**: Add checkpointer to `compile()` call
```python
# If running directly with Python (not langgraph dev)
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

**Fix 2**: Ensure you're passing thread_id in config
```python
# Thread ID must be in config, not state
config = {"configurable": {"thread_id": "conversation-123"}}
result = graph.invoke({"messages": [...]}, config=config)
```

---

### "Memory lost on server restart"

**Symptoms:**
- Conversations work during a session
- After restarting the server, all conversation history is lost
- Thread IDs don't retrieve previous conversations

**Cause**: Using MemorySaver in production (stores in RAM only)

**Why it happens:**
MemorySaver stores checkpoints in memory, which is cleared when the process restarts. For production, you need persistent storage.

**Fix**: Switch to PyMySQLSaver for persistent storage

```python
# Replace MemorySaver with PyMySQLSaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

checkpointer = PyMySQLSaver.from_conn_string(
    "mysql://user:password@host:port/database"
)
graph = workflow.compile(checkpointer=checkpointer)
```

**Setup MySQL:**
```bash
# 1. Install package
pip install langgraph-checkpoint-mysql pymysql

# 2. Create database
mysql -u root -p
CREATE DATABASE langgraph_db;

# 3. First run - call setup()
checkpointer.setup()  # Creates required tables
```

---

### MySQL Connection Errors

**Symptoms:**
- Certificate errors when connecting to MySQL
- "SSL connection error" or similar
- MySQL worked in old code structure but not in new structure

**Common causes:**
1. **Incorrect import** (see above - must use `PyMySQLSaver` not `MySQLSaver`)
2. **SSL/TLS certificate issues**
3. **Connection string format incorrect**
4. **Database not created or accessible**

**Fix for SSL/certificate errors:**
```python
# Add SSL parameters to connection string if needed
connection_string = (
    "mysql://user:password@host:port/database"
    "?ssl_ca=/path/to/ca.pem"
    "&ssl_verify_cert=true"
)
```

**Or disable SSL verification (not recommended for production):**
```python
connection_string = (
    "mysql://user:password@host:port/database"
    "?ssl_verify_cert=false"
)
```

**Verify connection string format:**
```python
# Correct format
mysql://username:password@hostname:port/database_name

# Examples
mysql://root:mypass@localhost:3306/langgraph_db
mysql://user:pass@db.example.com:3306/production_db
```

**Test connection independently:**
```python
import pymysql
try:
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='password',
        database='langgraph_db'
    )
    print("✓ MySQL connection successful!")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

---

## Summary

**Remember**:
- 🎨 **Studio** = No checkpointer (Studio has built-in)
- 🐍 **Python** = Yes checkpointer (you manage it)
- 🚀 **Production** = Yes checkpointer (PyMySQLSaver for persistence)

**Different contexts = different checkpointer usage!**

---

## See Also

- [README.md](README.md) - Full project documentation
- [graphs/multi_agent_system.py](graphs/multi_agent_system.py) - Studio example
- [main.py](main.py) - Direct Python example
- [api/server.py](api/server.py) - Production example
