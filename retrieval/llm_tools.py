"""
LLM API Tools for ChatIndex Retrieval

This module provides tools that LLMs can use to interact with a ChatIndex/CTree
for intelligent conversation retrieval.
"""

import json
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from ctree import CTree, TopicNode, MessageNode


# Tool definitions for LLM API
TOOLS = [
    {
        "name": "view_node_and_children",
        "description": "View a specific node in the ChatIndex tree and see its children. "
                      "Returns the node's topic name, summary, message range (start_index to end_index), "
                      "and a list of all direct children with their basic information. "
                      "Use this to navigate the conversation tree structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_path": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Path to the node as a list of child indices from root. "
                                 "Empty array [] means root node. [0] means first child of root. "
                                 "[0, 1] means second child of first child of root, etc."
                }
            },
            "required": ["node_path"]
        }
    },
    {
        "name": "get_node_messages",
        "description": "Retrieve the actual conversation messages covered by a node using its start and end indices. "
                      "Returns all messages in the range [start_index, end_index) from the conversation history. "
                      "Use this to read the actual content of a conversation segment after identifying it with view_node_and_children.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_index": {
                    "type": "integer",
                    "description": "Starting index in the conversation (inclusive)",
                    "minimum": 0
                },
                "end_index": {
                    "type": "integer",
                    "description": "Ending index in the conversation (exclusive)",
                    "minimum": 0
                }
            },
            "required": ["start_index", "end_index"]
        }
    }
]


class ChatIndexTools:
    """Handler for ChatIndex tool operations with LLM API."""

    def __init__(self, ctree: CTree):
        """
        Initialize ChatIndex tools with a CTree instance.

        Args:
            ctree: A CTree instance with conversation data
        """
        self.ctree = ctree

    def view_node_and_children(self, node_path: List[int]) -> Dict[str, Any]:
        """
        View a node and its children in the CTree.

        Args:
            node_path: List of child indices from root. Empty list for root.

        Returns:
            Dictionary with node information and children details
        """
        try:
            # Start from root
            current_node = self.ctree.root

            # Navigate to the target node
            for idx in node_path:
                if not hasattr(current_node, 'children') or idx >= len(current_node.children):
                    return {
                        "error": f"Invalid path: No child at index {idx}",
                        "valid_indices": list(range(len(current_node.children))) if hasattr(current_node, 'children') else []
                    }
                current_node = current_node.children[idx]

            # Build response
            result = {
                "node_path": node_path,
                "node_type": "topic" if isinstance(current_node, TopicNode) else "message",
            }

            if isinstance(current_node, TopicNode):
                result.update({
                    "topic_name": current_node.topic_name,
                    "summary": current_node.summary if current_node.summary else "No summary available",
                    "start_index": current_node.start_index,
                    "end_index": current_node.end_index,
                    "message_count": current_node.get_message_count(),
                    "child_count": current_node.sub_node_count,
                    "children": []
                })

                # Add children information
                for i, child in enumerate(current_node.children):
                    if isinstance(child, TopicNode):
                        result["children"].append({
                            "index": i,
                            "type": "topic",
                            "topic_name": child.topic_name,
                            "summary": child.summary if child.summary else "No summary",
                            "start_index": child.start_index,
                            "end_index": child.end_index,
                            "message_count": child.get_message_count(),
                            "child_count": child.sub_node_count
                        })
                    elif isinstance(child, MessageNode):
                        result["children"].append({
                            "index": i,
                            "type": "message",
                            "message_index": child.message_index,
                            "user_preview": child.user_message["content"][:100] + "..."
                                          if len(child.user_message["content"]) > 100
                                          else child.user_message["content"],
                            "assistant_preview": child.assistant_message["content"][:100] + "..."
                                               if len(child.assistant_message["content"]) > 100
                                               else child.assistant_message["content"]
                        })
            else:  # MessageNode
                result.update({
                    "message_index": current_node.message_index,
                    "user_message": current_node.user_message,
                    "assistant_message": current_node.assistant_message,
                    "system_message": current_node.system_message
                })

            return result

        except Exception as e:
            return {"error": f"Error viewing node: {str(e)}"}

    def get_node_messages(self, start_index: int, end_index: int) -> Dict[str, Any]:
        """
        Get messages from the conversation using start and end indices.

        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (exclusive)

        Returns:
            Dictionary with messages in the specified range
        """
        try:
            if start_index < 0 or end_index < 0:
                return {"error": "Indices must be non-negative"}

            if start_index >= end_index:
                return {"error": "start_index must be less than end_index"}

            total_messages = len(self.ctree.conversation)

            if start_index >= total_messages:
                return {
                    "error": f"start_index {start_index} exceeds conversation length {total_messages}"
                }

            # Clamp end_index to conversation length
            actual_end = min(end_index, total_messages)

            messages = self.ctree.conversation[start_index:actual_end]

            return {
                "start_index": start_index,
                "end_index": actual_end,
                "requested_end_index": end_index,
                "message_count": len(messages),
                "messages": messages
            }

        except Exception as e:
            return {"error": f"Error retrieving messages: {str(e)}"}

    def process_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a tool call from LLM.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "view_node_and_children":
            return self.view_node_and_children(tool_input["node_path"])
        elif tool_name == "get_node_messages":
            return self.get_node_messages(tool_input["start_index"], tool_input["end_index"])
        else:
            return {"error": f"Unknown tool: {tool_name}"}


def query_ctree(
    api_key: str,
    ctree: CTree,
    user_query: str,
    max_turns: int = 50
) -> Dict[str, Any]:
    """
    Query a CTree to answer questions about the conversation.

    Args:
        api_key: Anthropic API key
        ctree: CTree instance with conversation data
        user_query: User's question about the conversation
        max_turns: Maximum number of conversation turns

    Returns:
        Dictionary with the conversation history and final response
    """
    client = Anthropic(api_key=api_key)
    tools_handler = ChatIndexTools(ctree)

    # Initial system message explaining the context
    system_message = """You are an AI assistant with access to a ChatIndex tree structure containing a conversation history.

The conversation is organized hierarchically into topics and subtopics. You have two tools available:

1. view_node_and_children: Navigate the tree structure to understand topics and subtopics
2. get_node_messages: Retrieve actual message content from specific ranges

Start by viewing the root node to understand the conversation structure, then drill down into relevant topics to find the information needed to answer the user's query.

The tree uses a path-based navigation system where each node is accessed by a list of child indices from root:
- [] = root node
- [0] = first child of root
- [0, 1] = second child of the first child of root
- etc.

Each TopicNode has start_index and end_index fields that define the range of messages it covers."""

    messages = [
        {"role": "user", "content": user_query}
    ]

    conversation_history = []

    for turn in range(max_turns):
        # Make API call to LLM
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            system=system_message,
            tools=TOOLS,
            messages=messages
        )

        conversation_history.append({
            "turn": turn + 1,
            "response": response
        })

        # Check if we're done (no tool use)
        if response.stop_reason == "end_turn":
            # Extract final text response
            final_response = ""
            for block in response.content:
                if block.type == "text":
                    final_response += block.text

            return {
                "success": True,
                "final_response": final_response,
                "turns_used": turn + 1,
                "conversation_history": conversation_history
            }

        # Process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Process each tool use
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = tools_handler.process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, indent=2)
                    })

            # Add tool results to conversation
            messages.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # Unexpected stop reason
            return {
                "success": False,
                "error": f"Unexpected stop reason: {response.stop_reason}",
                "turns_used": turn + 1,
                "conversation_history": conversation_history
            }

    return {
        "success": False,
        "error": f"Reached maximum turns ({max_turns})",
        "conversation_history": conversation_history
    }


def query_ctree_streaming(
    api_key: str,
    ctree: CTree,
    user_query: str,
    max_turns: int = 50,
    on_text_chunk: callable = None,
    on_tool_use: callable = None,
    on_turn_complete: callable = None
) -> Dict[str, Any]:
    """
    Query a CTree with streaming output.

    Args:
        api_key: Anthropic API key
        ctree: CTree instance with conversation data
        user_query: User's question about the conversation
        max_turns: Maximum number of conversation turns
        on_text_chunk: Callback for text chunks (chunk_text: str)
        on_tool_use: Callback for tool use (tool_name: str, tool_input: dict)
        on_turn_complete: Callback when turn completes (turn_number: int)

    Returns:
        Dictionary with the conversation history and final response
    """
    client = Anthropic(api_key=api_key)
    tools_handler = ChatIndexTools(ctree)

    # Initial system message explaining the context
    system_message = """You are an AI assistant with access to a ChatIndex tree structure containing a conversation history.

The conversation is organized hierarchically into topics and subtopics. You have two tools available:

1. view_node_and_children: Navigate the tree structure to understand topics and subtopics
2. get_node_messages: Retrieve actual message content from specific ranges

Start by viewing the root node to understand the conversation structure, then drill down into relevant topics to find the information needed to answer the user's query.

The tree uses a path-based navigation system where each node is accessed by a list of child indices from root:
- [] = root node
- [0] = first child of root
- [0, 1] = second child of the first child of root
- etc.

Each TopicNode has start_index and end_index fields that define the range of messages it covers."""

    messages = [
        {"role": "user", "content": user_query}
    ]

    conversation_history = []
    final_response = ""

    for turn in range(max_turns):
        # Make streaming API call to LLM
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            system=system_message,
            tools=TOOLS,
            messages=messages
        ) as stream:
            current_content = []

            for event in stream:
                # Handle text delta events
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        if on_text_chunk:
                            on_text_chunk(event.delta.text)
                        final_response += event.delta.text

                # Handle content block start (for tool use)
                elif event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            current_content.append({
                                "type": "tool_use",
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": {}
                            })
                        elif event.content_block.type == "text":
                            current_content.append({
                                "type": "text",
                                "text": ""
                            })

                # Handle input_json_delta for tool use
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "partial_json"):
                        # Accumulate tool input JSON
                        pass

            # Get final message from stream
            final_message = stream.get_final_message()

            conversation_history.append({
                "turn": turn + 1,
                "response": final_message
            })

            if on_turn_complete:
                on_turn_complete(turn + 1)

            # Check if we're done (no tool use)
            if final_message.stop_reason == "end_turn":
                return {
                    "success": True,
                    "final_response": final_response,
                    "turns_used": turn + 1,
                    "conversation_history": conversation_history
                }

            # Process tool calls
            if final_message.stop_reason == "tool_use":
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })

                # Process each tool use
                tool_results = []
                for block in final_message.content:
                    if block.type == "tool_use":
                        if on_tool_use:
                            on_tool_use(block.name, block.input)

                        result = tools_handler.process_tool_call(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, indent=2)
                        })

                # Add tool results to conversation
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                # Unexpected stop reason
                return {
                    "success": False,
                    "error": f"Unexpected stop reason: {final_message.stop_reason}",
                    "turns_used": turn + 1,
                    "conversation_history": conversation_history
                }

    return {
        "success": False,
        "error": f"Reached maximum turns ({max_turns})",
        "conversation_history": conversation_history
    }
