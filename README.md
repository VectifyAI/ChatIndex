# ChatIndex - Tree Indexing for Long Conversations

ChatIndex is a context management system that enables LLMs to efficiently navigate and utilize long conversation histories through hierarchical tree-based indexing.

## Table of Contents

- [Motivation](#motivation)
- [ChatIndex Introduction](#chatindex-introduction)
  - [Inspiration & Comparisons](#inspiration--comparisons)
  - [Context Tree Specification](#context-tree-specification)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
  - [Tree Generation Example](#tree-generation-example)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Motivation

Current AI chat assistants face a fundamental challenge: **context management in long conversations**. While current LLM apps use multiple separate conversations to bypass context limits, a truly human-like AI assistant should maintain a single, coherent conversation thread, making efficient context management critical.

Although modern LLMs have longer contexts, they still suffer from the long-context problem (e.g. [context rot problem](https://research.trychroma.com/context-rot)) - reasoning ability decreases as context grows longer. Memory-based systems (e.g. [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952), [mem0](https://arxiv.org/pdf/2504.19413)) have been invented to alleviate the context rot problem, however, memory-based representations are inherently lossy and inevitably lose information from the original conversation. In principle, **no lossy representation is universally perfect** for all downstream tasks. This leads to two key requirements for defining a flexible in-context management system:

1. **Preserve raw data**: An index system that can retrieve the original conversation when necessary
2. **Multi-resolution access**: Ability to retrieve information at different levels of detail on-demand

## ChatIndex Introduction

**ChatIndex addresses this by providing hierarchical, dynamic-resolution information retrieval:**
- Preserves all raw conversation data
- Enables hierarchical retrieval: if a parent node contains sufficient information, child topics aren't retrieved
- Resolution is dynamic and problem-dependent

### Inspiration & Comparisons

ChatIndex is an extension of [PageIndex](https://pageindex.ai/blog/pageindex-intro), a tree-based index system for long documents, but adapted for conversational contexts with two key differences:

1. **Dynamic vs. Static**: 
   - Documents are fixed → tree generated once before downstream tasks
   - Conversations are dynamic → requires **incremental tree generation**

2. **Structured vs. Unstructured**:
   - Documents have natural structure (table of contents, sections)
   - Conversations are unstructured message lists → requires defining the structure within conversations.
   
Inspired by topic models (e.g. [LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), [HDP](http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/jasa2006.pdf)), ChatIndex uses LLMs to detect topic switches in long conversations and generate a tree with nodes that represent a topic. We call this a **Context Tree (CTree)**. Unlike hierarchical traditional topic models, the CTree is **temporally ordered** - new topics can only branch from the current topic or its ancestors.

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

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"

# or use a `.env` file:
echo "OPENAI_API_KEY=your-api-key-here" > .env
```


### Basic Usage

```python
from ctree import CTree

# Initialize tree (API key loaded from OPENAI_API_KEY env var or .env file)
tree = CTree(max_children=10)


# Add conversation exchanges
messages = [
    {"role": "system", "content": "You are a helpful programming tutor."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."}
]
tree.add(messages)

tree.print_tree()

```

### Tree Generation Example

To build a Context Tree from an existing conversation history, run the included demo:

```bash
python demo.py
```

See `demo.py` for the complete implementation details.

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

## Roadmap

- [ ] **Dynamic information retrieval**: Query-based context extraction from tree
- [ ] **Offline tree structure optimization**: Post-processing techniques to reorganize and optimize tree structure for better retrieval performance after initial tree construction 
- [ ] ...


## Contributing

This project is currently under active development. Any contributions are welcome! Please feel free to:
- Submit issues for bugs or feature requests
- Open pull requests with improvements
- Share your use cases and feedback
