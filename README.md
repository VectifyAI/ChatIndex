# ChatIndex - Tree Indexing for Long Conversations

ChatIndex is a context management system that enables LLMs to efficiently navigate and utilize long conversation histories through hierarchical tree-based indexing and intelligent reasoning-based retrieval.

## Table of Contents

- [Motivation](#motivation)
- [How It Works](#how-it-works)
- [ChatIndex Introduction](#chatindex-introduction)
  - [Inspiration & Comparisons](#inspiration--comparisons)
  - [Context Tree Specification](#context-tree-specification)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Complete Workflow](#complete-workflow)
  - [Phase 1: Building the Tree](#phase-1-building-the-tree)
  - [Phase 2: Querying the Tree](#phase-2-querying-the-tree)
- [Advanced Usage](#advanced-usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Motivation

Current AI chat assistants face a fundamental challenge: **context management in long conversations**. While current LLM apps use multiple separate conversations to bypass context limits, a truly human-like AI assistant should maintain a single, coherent conversation thread, making efficient context management critical.

Although modern LLMs have longer contexts, they still suffer from the long-context problem (e.g. [context rot problem](https://research.trychroma.com/context-rot)) - reasoning ability decreases as context grows longer. Memory-based systems (e.g. [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952), [mem0](https://arxiv.org/pdf/2504.19413)) have been invented to alleviate the context rot problem, however, memory-based representations are inherently lossy and inevitably lose information from the original conversation. In principle, **no lossy representation is universally perfect** for all downstream tasks. This leads to two key requirements for defining a flexible in-context management system:

1. **Preserve raw data**: An index system that can retrieve the original conversation when necessary
2. **Multi-resolution access**: Ability to retrieve information at different levels of detail on-demand

## ChatIndex Introduction
ChatIndex is designed to meet these requirements by constructing a hierarchical tree index—which we call a Context Tree (CTree)—that captures the structure and semantic organization of a long conversation.
Unlike memory-based architectures that store only compressed, lossy summaries, ChatIndex preserves the complete raw conversation and layers a topic hierarchy on top:
  * Leaf nodes store raw conversational segments.
  * Internal nodes store topic summaries that abstract and represent their child nodes.

This forms a multi-level topic hierarchy in which higher nodes represent broader themes and lower nodes convey increasingly specific details. See the figure below for an illustration.


<p align="center">
  <img width="600" alt="Root (1)" src="https://github.com/user-attachments/assets/5b0e5d2f-7486-43ad-9d86-251de1b6cbb0" />
</p>

When a query arrives, ChatIndex performs a top-down search through the topic tree. At each node, it evaluates whether the summary provides enough information for the query. If it does, the traversal stops and the system returns that higher-level summary; if not, it continues downward until more detailed information—or the raw conversation—is accessed. This design offers:

* Dynamic retrieval resolution – the system returns only as much detail as needed.
* Lossless fallback – the raw conversation is always accessible when required.
* Efficient reasoning – large contexts are reduced to the minimally sufficient subset.

By combining the completeness of an index system with the flexibility of a hierarchical memory structure, ChatIndex provides a scalable and robust solution for managing long conversational contexts.


### Inspiration & Comparisons

ChatIndex is an extension of [PageIndex](https://pageindex.ai/blog/pageindex-intro), a tree-based index system for long documents, but adapted for conversational contexts with two key differences:

1. **Dynamic vs. Static**: 
   - Documents are fixed → tree generated once before downstream tasks
   - Conversations are dynamic → requires **incremental tree generation**

2. **Structured vs. Unstructured**:
   - Documents have natural structure (table of contents, sections)
   - Conversations are unstructured message lists → requires defining the structure within conversations.
   
Inspired by topic models (e.g. [LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), [HDP](http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/jasa2006.pdf)), ChatIndex uses LLMs to detect topic switches in long conversations and generate a tree with nodes that represent a topic.  Unlike hierarchical traditional topic models, the CTree is **temporally ordered** - new topics can only branch from the current topic or its ancestors.


### Context Tree Specification

A Context Tree consists of two types of nodes:

#### 1. TopicNode
- Represents a conversation topic/subtopic
- Contains:
  - `topic_name`: Descriptive name (2-5 words)
  - `summary`: Brief summary of the topic content
  - `start_index`, `end_index`: Range of messages in the conversation
  - `children`: List of child TopicNodes or MessageNodes
  - `sub_node_count`: Number of direct children

#### 2. MessageNode
- Represents a leaf node containing an actual conversation exchange
- Contains:
  - `system_message`: Optional system message
  - `user_message`: User's message
  - `assistant_message`: Assistant's response
  - `message_index`: Position in the conversation history

#### Tree Structure Properties

- **Temporal ordering**: New topic nodes can only be children of the current node or its ancestors
- **Tree width control**: The maximum width of a tree layer is controlled by `max_children`, which guarantees that when conducting layer-wise tree search for relevant conversations, the context length will be controlled

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChatIndex.git
cd ChatIndex
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
# For building trees (Phase 1)
export OPENAI_API_KEY="your-openai-key"

# For querying trees (Phase 2)
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or use a .env file:
echo "OPENAI_API_KEY=your-openai-key" > .env
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
```

### Complete Workflow

Here's the full pipeline from conversation to intelligent retrieval:

```python
from ctree import CTree
from retrieval.llm_tools import query_ctree
import os

# ============================================
# Phase 1: Build the conversation tree
# ============================================
tree = CTree(max_children=10)

# Add your conversation messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    # ... more messages
]
tree.add(messages)

# Save for later use
tree.save('my_conversation.json')
tree.print_tree()

# ============================================
# Phase 2: Query the conversation tree
# ============================================
# Load the tree (can be done in a separate session)
tree = CTree.load('my_conversation.json')

# Ask questions about the conversation
result = query_ctree(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    ctree=tree,
    user_query="What programming concepts were discussed?",
    max_turns=50
)

print(result["final_response"])
print(f"Retrieved answer using {result['turns_used']} turns")
```

### Phase 1: Building the Tree

Build a hierarchical index of your conversation:

```python
from ctree import CTree

# Initialize tree
tree = CTree(max_children=10)

# Add conversation exchanges
messages = [
    {"role": "system", "content": "You are a helpful programming tutor."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."}
]
tree.add(messages)

# Save and visualize
tree.save('conversation_tree.json')
tree.print_tree()
```

### Tree Generation Example

To build a Context Tree from an existing conversation history, run the included demo:

```bash
python demo.py
```

See `demo.py` for the complete implementation details.

### Phase 2: Querying the Tree

Query your indexed conversation efficiently:

```python
from ctree import CTree
from retrieval.llm_tools import query_ctree
import os

# Load indexed conversation
tree = CTree.load('conversation_tree.json')

# Ask questions
result = query_ctree(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    ctree=tree,
    user_query="What topics were discussed about network protocols?",
    max_turns=50  # default is 50
)

print(result["final_response"])
```

**Key benefits:**
- **Cost reduction** - Only retrieves relevant conversation segments
- **Reasoning-based navigation** - LLM explores the tree autonomously
- **Multi-resolution** - Gets summaries or full messages as needed

#### Example Tree Output

Here's what a Context Tree structure looks like (from the demo):

```
ROOT
├── Network Protocols and Testing
│   ├── ICMP Message Types Comparison
│   │   └── [Message 0-2]
│   ├── Ping Command and Network Testing
│   │   ├── Ping Command and Basic Network Testing
│   │   │   ├── [Message 2-4]
│   │   │   ├── [Message 4-6]
│   │   │   └── [Message 6-10]
│   │   └── Advanced Network Diagnostics
│   │       └── [Message 10-12]
│   └── Network Protocol Analysis
│       └── [Message 12-16]
└── ... more topics
```

See `./save/conversation_tree.json` for a complete tree visualization

## Advanced Usage

### Streaming Responses

Get real-time responses while querying:

```python
from retrieval.llm_tools import query_ctree_streaming

def on_text(chunk):
    print(chunk, end='', flush=True)

def on_tool_use(tool_name, tool_input):
    print(f"\n[Using: {tool_name}]", flush=True)

result = query_ctree_streaming(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    ctree=tree,
    user_query="What are the main topics?",
    on_text_chunk=on_text,
    on_tool_use=on_tool_use
)
```

### Direct Tool Access

Use the tools directly without the LLM wrapper:

```python
from retrieval.llm_tools import ChatIndexTools

tools = ChatIndexTools(tree)

# Navigate the tree
root = tools.view_node_and_children([])  # View root
topic = tools.view_node_and_children([0])  # View first topic

# Get messages
messages = tools.get_node_messages(0, 10)  # Get messages 0-10
```

## Roadmap

- [x] **Hierarchical tree indexing** - Build topic-based conversation trees
- [x] **LLM-guided retrieval** - Intelligent navigation with tools
- [x] **Streaming support** - Real-time responses
- [ ] **Offline tree optimization** - Post-processing for better structure
- [ ] **Multi-LLM support** - Support for different LLMs in retrieval
- [ ] **Incremental updates** - Efficiently update trees with new messages
- [ ] **Vector search integration** - Hybrid retrieval combining tree + embeddings


## Contributing

This project is currently under active development. Any contributions are welcome! Please feel free to:
- Submit issues for bugs or feature requests
- Open pull requests with improvements
- Share your use cases and feedback
