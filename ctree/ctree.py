"""
Context Tree (CTree)
This module implements a hierarchical topic tree for conversation history,
where nodes represent topics and are organized in a temporally-ordered tree structure.
"""

import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from .utils import ChatGPT_API, extract_json

# Load environment variables from .env file
load_dotenv()


@dataclass(eq=False)
class Node:
    """
    Base class for all nodes in the conversation tree.
    
    Attributes:
        children: List of child nodes
        parent: Reference to parent node
        sub_node_count: Number of direct children (same as len(children))
    """
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None
    sub_node_count: int = 0
    
    def update_sub_node_count(self) -> None:
        """
        Update the sub_node_count for this node and all ancestors.
        sub_node_count represents the number of direct children.
        This should be called after adding a child to ensure counts are accurate.
        """
        # Count only direct children
        self.sub_node_count = len(self.children)
        
        # Propagate update to parent
        if self.parent is not None:
            self.parent.update_sub_node_count()
    
    def is_leaf(self) -> bool:
        """Returns True if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts node to dictionary representation."""
        raise NotImplementedError("Subclasses must implement to_dict()")


@dataclass(eq=False)
class MessageNode(Node):
    """
    Represents a message leaf node in the conversation tree.
    Each node contains a user message and an assistant response, optionally with a system message.
    
    This is a true leaf node - it cannot have children.
    
    Attributes:
        user_message: The user message dictionary
        assistant_message: The assistant message dictionary
        system_message: Optional system message dictionary
        message_index: Index of this message in the conversation
    """
    user_message: Dict = field(default_factory=dict)
    assistant_message: Dict = field(default_factory=dict)
    system_message: Optional[Dict] = None
    message_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts message node to dictionary representation."""
        result = {
            "type": "message",
            "message_index": self.message_index,
            "user": self.user_message.get("content", "")[:200],
            "assistant": self.assistant_message.get("content", "")[:200],
            "sub_node_count": self.sub_node_count  # Always 0 for leaf nodes
        }
        if self.system_message:
            result["system"] = self.system_message.get("content", "")[:200]
        return result
    
    def __repr__(self) -> str:
        user_preview = self.user_message.get("content", "")[:40]
        assistant_preview = self.assistant_message.get("content", "")[:40]
        if self.system_message:
            system_preview = self.system_message.get("content", "")[:40]
            return f"MessageNode([{self.message_index}]: S:{system_preview}... U:{user_preview}... A:{assistant_preview}...)"
        return f"MessageNode([{self.message_index}]: U:{user_preview}... A:{assistant_preview}...)"


@dataclass(eq=False)
class TopicNode(Node):
    """
    Represents a topic node in the conversation tree.
    
    Attributes:
        topic_name: Name of the topic
        summary: Summary of the topic content
        start_index: Starting index in conversation history (inclusive)
        end_index: Ending index in conversation history (exclusive)
    """
    topic_name: str = ""
    summary: str = ""
    start_index: int = 0
    end_index: int = 0
    
    def get_message_count(self) -> int:
        """Returns the number of messages covered by this node (including descendants)."""
        if self.is_leaf():
            return self.end_index - self.start_index
        # Count MessageNode children recursively
        count = 0
        for child in self.children:
            if isinstance(child, MessageNode):
                count += 1
            elif isinstance(child, TopicNode):
                count += child.get_message_count()
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts topic node to dictionary representation."""
        return {
            "type": "topic",
            "topic_name": self.topic_name,
            "summary": self.summary,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "sub_node_count": self.sub_node_count,
            "children": [child.to_dict() for child in self.children]
        }
    
    def __repr__(self) -> str:
        return f"TopicNode('{self.topic_name}', [{self.start_index}:{self.end_index}], {self.sub_node_count} children)"


class CTree:
    """
    Context Tree - A hierarchical topic tree for conversation history.
    
    The tree maintains temporal ordering where new nodes can only be children
    of the current node or its ancestors.
    """
    
    def __init__(self, max_children: int = 5, api_key: Optional[str] = None, model: str = "gpt-4o-mini", auto_save_path: Optional[str] = None):
        """
        Initialize the CTree.
        
        Args:
            max_children: Maximum number of direct children (messages or topics) before reorganizing a node.
                         - If a node has > max_children message children → expand into subtopics (vertical split)
                         - If a node has > max_children topic children → split into two siblings (horizontal split)
                         - If ROOT has > max_children topic children → expand into subtopics (parent remains root)
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
            model: LLM model to use for topic generation and classification
            auto_save_path: Optional path to automatically save the tree after each message addition.
                           If provided, the tree will be saved incrementally to prevent data loss.
        """
        self.max_children = max_children
        self.model = model
        self.conversation: List[Dict] = []
        self.auto_save_path = auto_save_path
        
        # Store API key for ChatGPT_API calls
        if api_key:
            self.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY  environment variable")
        
        # Create virtual root node
        self.root = TopicNode(
            topic_name="ROOT",
            summary="Virtual root node of the conversation tree",
            start_index=0,
            end_index=0
        )
        self.current_node = self.root
    
    def get_ancestors(self, node: TopicNode, include_self: bool = True, exclude_root: bool = False) -> List[TopicNode]:
        """
        Get all ancestor nodes of a given node.
        
        Args:
            node: The node to find ancestors for
            include_self: If True, includes the node itself (Anc*(v))
            exclude_root: If True, excludes the ROOT node from the result
        
        Returns:
            List of ancestor nodes from root (or first non-root) to node
        """
        ancestors = []
        current = node if include_self else node.parent
        
        while current is not None:
            ancestors.insert(0, current)  # Insert at beginning to maintain order
            current = current.parent
        
        # Filter out root if requested
        if exclude_root:
            ancestors = [n for n in ancestors if n.topic_name != "ROOT"]
        
        return ancestors
    
    def _is_frozen_node(self, node: TopicNode) -> bool:
        """
        Check if a node is frozen (won't change anymore).
        A node is frozen if it's not the current_node and not an ancestor of current_node.
        
        Args:
            node: The node to check
        
        Returns:
            True if the node is frozen, False otherwise
        """
        if node is self.current_node:
            return False
        
        # Check if node is an ancestor of current_node
        ancestors = self.get_ancestors(self.current_node, include_self=False, exclude_root=False)
        return node not in ancestors
    
    def _generate_summaries_for_frozen_nodes(self, node: Optional[Node] = None) -> None:
        """
        Generate summaries for all frozen nodes (nodes that won't change anymore).
        This traverses the tree and generates summaries only for TopicNodes that are frozen.
        
        Args:
            node: Node to start from (defaults to root)
        """
        if node is None:
            node = self.root
        
        # Only process TopicNodes
        if isinstance(node, TopicNode):
            # Skip root node
            if node is not self.root and self._is_frozen_node(node):
                # Only generate summary if it's empty or not set
                if not node.summary or node.summary.strip() == "":
                    # Get messages for this node
                    messages = self.conversation[node.start_index:node.end_index]
                    if messages:
                        node.summary = self._llm_summarize(messages, node.topic_name)
        
        # Recursively process children
        if hasattr(node, 'children'):
            for child in node.children:
                self._generate_summaries_for_frozen_nodes(child)
    
    def add(self, messages: List[Dict]) -> None:
        """Add a message to the conversation."""
        # Parse messages by role into a clean dict structure
        msg_dict = {
            "system": next((m for m in messages if m.get("role") == "system"), None),
            "user": next((m for m in messages if m.get("role") == "user"), None),
            "assistant": next((m for m in messages if m.get("role") == "assistant"), None),
            "index": len(self.conversation)
        }
        # Validate required messages exist
        if not msg_dict["user"] or not msg_dict["assistant"]:
            raise ValueError("Exchange must contain user and assistant messages")
        
        # Add all messages to conversation history
        for msg in messages:
            self.conversation.append(msg)
        
        # First exchange - initialize tree
        if len(self.root.children) == 0:
            self._initialize_first_topic_with_message(msg_dict)
        else:
            # Add exchange to existing tree
            self._add_message(msg_dict)
        
        # Auto-save if path is configured (always save conversation for auto-save)
        if self.auto_save_path:
            self.save(self.auto_save_path, save_conversation=True)
    
    
    def _initialize_first_topic_with_message(self, msg_dict: Dict) -> None:
        """Initialize the tree with the first message."""
        # Generate topic from the first exchange
        topic_name = self._llm_generate_topic_from_message(
            msg_dict["user"], 
            msg_dict["assistant"], 
            system_msg=msg_dict["system"]
        )
        
        # Determine message count (2 or 3)
        msg_count = 3 if msg_dict["system"] else 2
        
        # Create the first topic node
        first_topic = TopicNode(
            topic_name=topic_name,
            start_index=0,
            end_index=msg_count,
            parent=self.root
        )
        
        # Create a message node (leaf node)
        msg_node = MessageNode(
            user_message=msg_dict["user"],
            assistant_message=msg_dict["assistant"],
            system_message=msg_dict["system"],
            message_index=msg_dict["index"],
            parent=first_topic
        )
        
        # Add message as child of the first topic
        first_topic.children.append(msg_node)
        
        # Add first topic to root
        self.root.children.append(first_topic)
        self.root.update_sub_node_count()
        self.root.end_index = msg_count
        self.current_node = first_topic
    
    def _add_message(self, msg_dict: Dict) -> None:
        """Add a message to the tree."""
        # Determine which topic this message belongs to
        target_topic = self._assign_topic_for_message(
            msg_dict["user"], 
            msg_dict["assistant"], 
            msg_dict["system"]
        )
        
        # Create a MessageNode for this message (leaf node)
        msg_node = MessageNode(
            user_message=msg_dict["user"],
            assistant_message=msg_dict["assistant"],
            system_message=msg_dict["system"],
            message_index=msg_dict["index"],
            parent=target_topic
        )
        
        # Add the message node as a child of the topic
        target_topic.children.append(msg_node)
        target_topic.update_sub_node_count()
        
        # Update the end_index of the target topic
        target_topic.end_index = len(self.conversation)
        self.current_node = target_topic
        
        # Check if any nodes need reorganization due to too many children
        self._check_and_reorganize_nodes()
        
        # Generate summaries for frozen nodes (nodes that won't change anymore)
        # self._generate_summaries_for_frozen_nodes()
    
    def _assign_topic_for_message(self, user_msg: Dict, assistant_msg: Dict, system_msg: Optional[Dict] = None) -> TopicNode:
        """
        Assign a message to a topic using LLM classification.
        
        Args:
            user_msg: The user message
            assistant_msg: The assistant message
            system_msg: Optional system message
        
        Returns:
            The TopicNode that should contain this message
        """
        # Get ancestors including current node (Anc*(v_k))
        candidate_nodes = self.get_ancestors(self.current_node, include_self=True, exclude_root=False)
        
        if not candidate_nodes:
            # Create new topic under root
            return self._create_new_topic_for_message(user_msg, assistant_msg, self.root, "", system_msg)
        
        # Use LLM to classify the message
        classification = self._llm_classify_message_exchange(user_msg, assistant_msg, candidate_nodes, system_msg)
        
        
        if classification["belongs_to_current"]:
            # Message belongs to current node (last in candidate_nodes)
            return candidate_nodes[-1]
        else:
            # Create new topic
            parent_index = classification.get("new_topic_parent_index", len(candidate_nodes) - 1)
            parent_node = candidate_nodes[parent_index]
            topic_name = classification.get("new_topic_name", "")
            
            return self._create_new_topic_for_message(user_msg, assistant_msg, parent_node, topic_name, system_msg)
    
    def _has_topic_children(self, node: TopicNode) -> bool:
        """Check if a node has any TopicNode children (vs only MessageNode children)."""
        return any(isinstance(child, TopicNode) for child in node.children)
    
    def _assign_topic(self, message: Dict, message_index: int) -> TopicNode:
        """
        Assign a message to a topic using LLM classification.
        
        Args:
            message: The message to assign
            message_index: Index of the message in conversation history
        
        Returns:
            The node that should contain this message
        """
        # Get ancestors including current node (Anc*(v_k)), excluding root
        candidate_nodes = self.get_ancestors(self.current_node, include_self=True, exclude_root=False)
        
        if not candidate_nodes:
            # Create new topic under root
            return self._create_new_node(message, message_index, self.root)
        
        # Use LLM to classify
        classification = self._llm_classify_message(message, candidate_nodes)
        
        if classification["belongs_to_current"]:
            # Message belongs to current node (last in candidate_nodes)
            return candidate_nodes[-1]
        else:
            # Create new topic
            parent_index = classification.get("new_topic_parent_index", len(candidate_nodes) - 1)
            parent_node = candidate_nodes[parent_index]
            topic_name = classification.get("new_topic_name", "")
            
            return self._create_new_node(message, message_index, parent_node, topic_name)
    
    def _create_new_node(self, message: Dict, start_index: int, parent: TopicNode, topic_name: str = "") -> TopicNode:
        """
        Create a new topic node for a message.
        
        Args:
            message: The message to create node for
            start_index: Starting index in conversation
            parent: Parent node
            topic_name: Optional topic name (if empty, will generate using LLM)
        
        Returns:
            The newly created TopicNode (note: does NOT add the message as a child yet)
        """
        # If no topic name provided, generate one
        if not topic_name or topic_name.strip() == "":
            topic_name = self._llm_generate_topic(message, parent)
        
        # Create new topic node
        new_topic = TopicNode(
            topic_name=topic_name,
            # summary=self._llm_summarize([message], topic_name),
            start_index=start_index,
            end_index=start_index,  # Will be updated when message is added
            parent=parent
        )
        
        # Add to parent's children
        parent.children.append(new_topic)
        parent.update_sub_node_count()
        
        return new_topic
    
    def _create_new_topic_for_message(self, user_msg: Dict, assistant_msg: Dict, parent: TopicNode, topic_name: str = "", system_msg: Optional[Dict] = None) -> TopicNode:
        """
        Create a new topic node for a message.
        
        Args:
            user_msg: The user message
            assistant_msg: The assistant message
            parent: Parent node
            topic_name: Optional topic name (if empty, will generate using LLM)
            system_msg: Optional system message
        
        Returns:
            The newly created TopicNode
        """
        # If no topic name provided, generate one from the message
        if not topic_name or topic_name.strip() == "":
            topic_name = self._llm_generate_topic_from_message(user_msg, assistant_msg, parent=parent, system_msg=system_msg)
        
        # Determine message count (account for system message if present)
        msg_count = 3 if system_msg else 2
        
        # Create new topic node
        new_topic = TopicNode(
            topic_name=topic_name,
            # summary=self._llm_summarize([user_msg, assistant_msg], topic_name),
            start_index=len(self.conversation) - msg_count,  # Start of this message
            end_index=len(self.conversation),  # End of this message
            parent=parent
        )
        
        # Add to parent's children
        parent.children.append(new_topic)
        parent.update_sub_node_count()
        
        return new_topic
    
    def _expand_node(self, node: TopicNode) -> None:
        """
        Expand a node into subtopics when it has too many message children.
        
        This reorganizes the MessageNode children into subtopic groups (vertical split).
        
        Args:
            node: The node to expand
        """
        # Get all message nodes from this topic (should only have MessageNode children at this point)
        msg_nodes = [child for child in node.children if isinstance(child, MessageNode)]
        
        if not msg_nodes:
            return  # Nothing to expand
        
        # Get the actual messages from nodes (flatten user + assistant for LLM analysis)
        messages = []
        for msg_node in msg_nodes:
            messages.append(msg_node.user_message)
            messages.append(msg_node.assistant_message)
        
        # Use LLM to identify subtopics
        subtopics = self._llm_split_subtopics(messages, node)
        
        # Clear current children (we'll reorganize them)
        node.children.clear()
        
        # Create child topic nodes for each subtopic and assign messages to them
        last_subtopic_node = None
        for subtopic in subtopics:
            start_offset = subtopic["start_offset"]
            end_offset = subtopic["end_offset"]
            
            # Since messages are exchanges (user+assistant), convert to message indices
            # Each message = 2 conversation messages, so divide by 2
            start_msg_idx = start_offset // 2
            end_msg_idx = (end_offset + 1) // 2
            
            # Create subtopic node
            subtopic_node = TopicNode(
                topic_name=subtopic["topic_name"],
                # summary=subtopic["summary"],
                start_index=node.start_index + start_offset,
                end_index=node.start_index + end_offset,
                parent=node
            )
            
            # Move relevant message nodes to this subtopic
            for i in range(start_msg_idx, min(end_msg_idx, len(msg_nodes))):
                msg_node = msg_nodes[i]
                msg_node.parent = subtopic_node
                subtopic_node.children.append(msg_node)
            
            node.children.append(subtopic_node)
            last_subtopic_node = subtopic_node
        
        # Update sub-node count after reorganization
        node.update_sub_node_count()
        
        # Update current_node if it was the expanded node
        if self.current_node is node and last_subtopic_node is not None:
            self.current_node = last_subtopic_node  # Point to the last (most recent) subtopic
    
    def _llm_generate_topic(self, message: Dict, parent: Optional[TopicNode] = None) -> str:
        """
        Generate a topic name for a message using LLM.
        
        Args:
            message: The message to generate topic for
            parent: Parent node for context
        
        Returns:
            Generated topic name
        """
        content = message.get("content", "")
        
        context = ""
        if parent and parent.topic_name != "ROOT":
            context = f"\nParent topic: {parent.topic_name}"
        
        prompt = f"""Given the following message, generate a concise topic name (2-5 words) that captures its main subject.{context}

Message: {content}

Respond with ONLY the topic name, nothing else."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.3, max_tokens=50)
            return response.strip()
        except Exception as e:
            print(f"LLM error in topic generation: {e}")
            return f"Topic at index {len(self.conversation)}"
    
    def _llm_generate_topic_from_message(self, user_msg: Dict, assistant_msg: Dict, parent: Optional[TopicNode] = None, system_msg: Optional[Dict] = None) -> str:
        """
        Generate a topic name for a message using LLM.
        
        Args:
            user_msg: The user message
            assistant_msg: The assistant message
            parent: Parent node for context
            system_msg: Optional system message
        
        Returns:
            Generated topic name
        """
        user_content = user_msg.get("content", "")
        assistant_content = assistant_msg.get("content", "")
        
        context = ""
        if parent and parent.topic_name != "ROOT":
            context = f"\nParent topic: {parent.topic_name}"
        
        # Include system message if present
        system_context = ""
        if system_msg:
            system_content = system_msg.get("content", "")
            system_context = f"\nSystem: {system_content}\n"
        
        prompt = f"""Given the following conversation exchange, generate a concise topic name (2-5 words) that captures its main subject.{context}
{system_context}
User: {user_content}
Assistant: {assistant_content}

Respond with ONLY the topic name, nothing else."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.3, max_tokens=50)
            return response.strip()
        except Exception as e:
            print(f"LLM error in topic generation from message: {e}")
            return f"Topic at message {len(self.conversation) // 2}"
    
    def _llm_summarize(self, messages: List[Dict], topic_name: str) -> str:
        """
        Generate a summary for a set of messages.
        
        Args:
            messages: List of messages to summarize
            topic_name: The topic name
        
        Returns:
            Generated summary
        """
        # Combine message contents
        content = "\n\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages[:5]  # Limit to first 5 messages for summary
        ])
        
        prompt = f"""Summarize the following conversation segment about "{topic_name}" in 1-2 sentences.

{content}

Summary:"""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.3, max_tokens=100)
            return response.strip()
        except Exception as e:
            print(f"LLM error in summarization: {e}")
            return f"Discussion about {topic_name}"
    
    def _llm_classify_message(self, message: Dict, candidate_nodes: List[TopicNode]) -> Dict[str, Any]:
        """
        Classify whether a message belongs to the current topic or starts a new one.
        
        Args:
            message: The message to classify
            candidate_nodes: List of candidate ancestor nodes (current node is last)
        
        Returns:
            Dictionary with classification result:
            {
                "reasoning": str,
                "belongs_to_current": bool,
                "new_topic_name": str (if creating new topic),
                "parent_index": int (index in candidate_nodes)
            }
        """
        content = message.get("content", "")
        
        # Current node is the last one in candidate_nodes
        current_node = candidate_nodes[-1]
        
        # Get recent messages from current node
        recent_messages = self.conversation[max(current_node.start_index, current_node.end_index - 3):current_node.end_index]
        recent_content = "\n".join([
            f"- {m.get('role', 'unknown')}: {m.get('content', '')[:150]}"
            for m in recent_messages[-3:]  # Last 3 messages
        ])
        
        # Build ancestor context for potential parent selection
        ancestor_list = "\n".join([
            f"{i}. {node.topic_name} - {node.summary[:100]}"
            for i, node in enumerate(candidate_nodes[:-1])
        ])
        
        prompt = f"""Analyze if this new message belongs to the current topic or should start a new topic.

**Current Topic:** {current_node.topic_name}
**Topic Summary:** {current_node.summary}

**Recent messages in current topic:**
{recent_content}

**New message:**
{content}

**Parent candidates (for parent selection if creating new topic):**
{ancestor_list}

Think step by step:
1. Does this message continue discussing the current topic?
2. If not, what new topic does it introduce?
3. Which ancestor topic should be the parent of this new topic?

Respond ONLY with valid JSON in this exact format:
{{
  "reasoning": "brief explanation of your decision",
  "belongs_to_current": true or false,
  "new_topic_name": "name of new topic (only if belongs_to_current is false, otherwise N/A)",
  "new_topic_parent_index": <only if belongs_to_current is false, otherwise N/A, index of parent topic, choose from the parent candidates list>
}}

Directly output ONLY the JSON, do not include any other text."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key)
            result = extract_json(response)
            
            # Validate and bound parent_index
            if "parent_index" in result:
                result["parent_index"] = min(max(0, int(result["parent_index"])), len(candidate_nodes) - 1)
            else:
                result["parent_index"] = len(candidate_nodes) - 1
                
            # Ensure belongs_to_current is boolean
            if "belongs_to_current" not in result:
                result["belongs_to_current"] = True
            
            return result
        except Exception as e:
            print(f"LLM error in classification: {e}")
            if 'response' in locals():
                print(f"Response was: {response}")
            # Default: continue current topic
            return {
                "reasoning": "Error in classification, defaulting to current topic",
                "belongs_to_current": True,
                "new_topic_name": "",
                "parent_index": len(candidate_nodes) - 1
            }
    
    def _llm_classify_message_exchange(self, user_msg: Dict, assistant_msg: Dict, candidate_nodes: List[TopicNode], system_msg: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify whether a message belongs to the current topic or starts a new one.
        
        Args:
            user_msg: The user message
            assistant_msg: The assistant message
            candidate_nodes: List of candidate ancestor nodes (current node is last)
            system_msg: Optional system message
        
        Returns:
            Dictionary with classification result
        """
        user_content = user_msg.get("content", "")
        assistant_content = assistant_msg.get("content", "")
        
        # Current node is the last one in candidate_nodes
        current_node = candidate_nodes[-1]
        
        # Get recent messages from current node
        recent_messages = self.conversation[max(0, current_node.end_index - 6):current_node.end_index]
        recent_content = "\n".join([
            f"- {m.get('role', 'unknown')}: {m.get('content', '')[:150]}"
            for m in recent_messages[-6:]  # Last 3 exchanges (6 messages)
        ])
        
        # Build ancestor context for potential parent selection
        ancestor_list = "\n".join([
            f"{i}. {node.topic_name} - {node.summary[:100]}"
            for i, node in enumerate(candidate_nodes[:-1])
        ])
        
        # Include system message if present
        system_context = ""
        if system_msg:
            system_content = system_msg.get("content", "")
            system_context = f"System: {system_content[:200]}\n"
        
        prompt = f"""Analyze if this new conversation exchange belongs to the current topic or should start a new topic.

**Current Topic:** {current_node.topic_name}
**Topic Summary:** {current_node.summary}

**Recent exchanges in current topic:**
{recent_content}

**New exchange:**
{system_context}User: {user_content[:400]}
Assistant: {assistant_content[:400]}

**Parent candidates (for parent selection if creating new topic):**
{ancestor_list}

Think step by step:
1. Does this exchange continue discussing the current topic?
2. If not, what new topic does it introduce?
3. Which ancestor topic should be the parent of this new topic?

Respond ONLY with valid JSON in this exact format:
{{
  "reasoning": "brief explanation of your decision",
  "belongs_to_current": true or false,
  "new_topic_name": "name of new topic (only if belongs_to_current is false, otherwise N/A)",
  "new_topic_parent_index": <only if belongs_to_current is false, otherwise N/A, index of parent topic>
}}

Directly output ONLY the JSON, do not include any other text."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.1, max_tokens=200)
            result = extract_json(response)
            
            # Validate and bound parent_index
            if "new_topic_parent_index" in result:
                if result["new_topic_parent_index"] != "N/A":
                    result["new_topic_parent_index"] = min(max(0, int(result["new_topic_parent_index"])), len(candidate_nodes) - 1)
                
            # Ensure belongs_to_current is boolean
            if "belongs_to_current" not in result:
                result["belongs_to_current"] = True
            
            return result
        except Exception as e:
            print(f"LLM error in message classification: {e}")
            if 'response' in locals():
                print(f"Response was: {response}")
            # Default: continue current topic
            return {
                "reasoning": "Error in classification, defaulting to current topic",
                "belongs_to_current": True,
                "new_topic_name": "",
                "new_topic_parent_index": len(candidate_nodes) - 1
            }
    
    def _llm_split_subtopics(self, messages: List[Dict], node: TopicNode) -> List[Dict[str, Any]]:
        """
        Split a node's messages into subtopics using LLM.
        
        Args:
            messages: List of messages in the node
            node: The node being split
        
        Returns:
            List of subtopic dictionaries
        """
        # Create a condensed representation of messages
        message_summary = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            message_summary.append(f"[{i}] {role}: {content}")
        
        messages_text = "\n".join(message_summary)
        
        prompt = f"""Analyze the following conversation segment about "{node.topic_name}" and identify distinct subtopics.
Group consecutive messages into 2 coherent subtopics.

Messages (indices 0-{len(messages)-1}):
{messages_text}

Respond in JSON format with an array of subtopics:
{{
  "subtopics": [
    {{
      "topic_name": "subtopic name",
      "summary": "brief summary",
      "start_offset": <start index>,
      "end_offset": <end index (exclusive)>
    }},
    ...
  ]
}}

Ensure:
- Subtopics cover all messages (start_offset of first is 0, end_offset of last is {len(messages)})
- No gaps or overlaps between subtopics
- Each subtopic has at least 2 messages

Directly output ONLY the JSON, do not include any other text."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.3, max_tokens=500)
            result = extract_json(response)
            
            # Handle both array and object with array responses
            if isinstance(result, list):
                subtopics = result
            elif "subtopics" in result:
                subtopics = result["subtopics"]
            else:
                # Fallback: create simple split
                mid = len(messages) // 2
                return [
                    {
                        "topic_name": f"{node.topic_name} - Part 1",
                        "summary": "First part of discussion",
                        "start_offset": 0,
                        "end_offset": mid
                    },
                    {
                        "topic_name": f"{node.topic_name} - Part 2",
                        "summary": "Second part of discussion",
                        "start_offset": mid,
                        "end_offset": len(messages)
                    }
                ]
            
            return subtopics
        except Exception as e:
            print(f"LLM error in splitting: {e}")
            # Fallback: split in half
            mid = len(messages) // 2
            return [
                {
                    "topic_name": f"{node.topic_name} - Part 1",
                    "summary": "First part of discussion",
                    "start_offset": 0,
                    "end_offset": mid
                },
                {
                    "topic_name": f"{node.topic_name} - Part 2",
                    "summary": "Second part of discussion",
                    "start_offset": mid,
                    "end_offset": len(messages)
                }
            ]
    
    def _check_and_reorganize_nodes(self) -> None:
        """
        Check all nodes in the tree to see if any have too many direct children.
        
        Reorganization strategy based on child type:
        - If node has > max_children message children → expand into subtopics (vertical)
        - If node has > max_children topic children → split into siblings (horizontal)
        """
        # Collect nodes that need reorganization (can't modify tree during traversal)
        nodes_to_expand = []   # Nodes with too many message children
        nodes_to_split = []    # Nodes with too many topic children
        
        def check_node(node: Node) -> None:
            """Recursively check nodes for reorganization."""
            if isinstance(node, TopicNode) and len(node.children) > self.max_children:
                # Determine type of children
                has_topic_children = self._has_topic_children(node)
                
                if has_topic_children:
                    # Has topic children → need to split horizontally
                    topic_children_count = len([c for c in node.children if isinstance(c, TopicNode)])
                    if topic_children_count > self.max_children:
                        nodes_to_split.append(node)
                else:
                    message_children_count = len([c for c in node.children if isinstance(c, MessageNode)])
                    if message_children_count > self.max_children:
                        nodes_to_expand.append(node)
            
            # Recursively check children
            for child in node.children:
                check_node(child)
        
        # Start checking from root
        check_node(self.root)
        
        # Expand nodes with too many message children
        for node in nodes_to_expand:
            self._expand_node(node)
        
        # Split nodes with too many topic children
        for node in nodes_to_split:
            self._split_node(node)
    
    def _expand_root_into_subtopics(self, root_node: TopicNode) -> None:
        """
        Expand root node into hierarchical subtopics when it has too many topic children.
        
        This creates intermediate topic layers under root to group related topics together.
        The parent of the new intermediate nodes remains the root.
        
        Args:
            root_node: The root node to expand
        """
        # Get all TopicNode children from root
        topic_children = [child for child in root_node.children if isinstance(child, TopicNode)]
        
        if len(topic_children) < 2:
            # Need at least 2 topic children to group
            return
        
        # Use LLM to group topics into logical subtopics
        subtopic_groups = self._llm_group_topics_into_subtopics(topic_children, root_node)
        
        # Clear current children (we'll reorganize them)
        root_node.children.clear()
        
        # Create intermediate topic nodes for each group
        last_subtopic_node = None
        for group in subtopic_groups:
            group_topic_indices = group["topic_indices"]
            
            # Get the topics for this group
            group_topics = [topic_children[i] for i in group_topic_indices if i < len(topic_children)]
            
            if not group_topics:
                continue
            
            # Determine index ranges for this group
            group_start = min(t.start_index for t in group_topics)
            group_end = max(t.end_index for t in group_topics)
            
            # Create intermediate subtopic node
            subtopic_node = TopicNode(
                topic_name=group["topic_name"],
                summary=group.get("summary", ""),
                start_index=group_start,
                end_index=group_end,
                parent=root_node
            )
            
            # Move topics to this subtopic
            for topic in group_topics:
                topic.parent = subtopic_node
                subtopic_node.children.append(topic)
            
            # Update sub-node count for the subtopic
            subtopic_node.update_sub_node_count()
            
            # Add subtopic to root
            root_node.children.append(subtopic_node)
            last_subtopic_node = subtopic_node
        
        # Update root's sub-node count
        root_node.update_sub_node_count()
        
        # Update current_node if needed - point to the most recent subtopic
        if last_subtopic_node is not None:
            # Find which subtopic contains the current node
            def find_node_in_subtree(node: Node, target: Node) -> bool:
                if node is target:
                    return True
                for child in node.children:
                    if find_node_in_subtree(child, target):
                        return True
                return False
            
            # Check which subtopic contains the current node
            for subtopic in root_node.children:
                if isinstance(subtopic, TopicNode) and find_node_in_subtree(subtopic, self.current_node):
                    # Current node is in this subtopic, no need to change
                    break
            else:
                # Current node not found, point to last subtopic's last child
                self.current_node = last_subtopic_node
        
    
    def _split_node(self, node: TopicNode) -> None:
        """
        Split a node into two sibling nodes when it has too many children.
        
        For root node: expand into sub-topics (parent remains root)
        For non-root nodes: split into sibling nodes
        
        Args:
            node: The node to split (must be a TopicNode)
        """
        if node is self.root:
            # Special handling for root node - expand into subtopics instead of splitting
            self._expand_root_into_subtopics(node)
            return
        
        parent = node.parent
        if parent is None:
            return
        
        # Get only TopicNode children (we split based on topic children, not messages)
        topic_children = [child for child in node.children if isinstance(child, TopicNode)]
        
        if len(topic_children) < 2:
            # Need at least 2 topic children to split
            return
        
        
        # Use LLM to find split point
        split_point = self._llm_find_split_point(topic_children, node)
        
        
        # Create two new nodes
        first_half_topics = topic_children[:split_point]
        second_half_topics = topic_children[split_point:]
        
        # Determine index ranges
        first_start = first_half_topics[0].start_index
        first_end = first_half_topics[-1].end_index
        second_start = second_half_topics[0].start_index
        second_end = second_half_topics[-1].end_index
        
        # Get messages for each half (for summary generation)
        first_messages = self.conversation[first_start:first_end]
        second_messages = self.conversation[second_start:second_end]
        
        # Generate names and summaries for the two new nodes
        first_name = self._llm_generate_topic_from_children(first_half_topics, node)
        second_name = self._llm_generate_topic_from_children(second_half_topics, node)
        
        # Create first split node
        first_node = TopicNode(
            topic_name=first_name,
            # summary=self._llm_summarize(first_messages[:5], first_name),
            start_index=first_start,
            end_index=first_end,
            parent=parent
        )
        
        # Create second split node
        second_node = TopicNode(
            topic_name=second_name,
            # summary=self._llm_summarize(second_messages[:5], second_name),
            start_index=second_start,
            end_index=second_end,
            parent=parent
        )
        
        # Move topic children to new nodes
        for topic in first_half_topics:
            topic.parent = first_node
            first_node.children.append(topic)
        
        for topic in second_half_topics:
            topic.parent = second_node
            second_node.children.append(topic)
        
        # Remove old node from parent and add new nodes
        parent.children.remove(node)
        parent.children.extend([first_node, second_node])
        
        # Update sub-node counts
        first_node.update_sub_node_count()
        second_node.update_sub_node_count()
        parent.update_sub_node_count()
        
        # Update current_node if it was the split node
        if self.current_node is node:
            self.current_node = second_node  # Point to the more recent half
        
    
    def _llm_find_split_point(self, topic_children: List[TopicNode], parent_node: TopicNode) -> int:
        """
        Use LLM to find the optimal split point in a list of topic children.
        
        Args:
            topic_children: List of TopicNode children to split
            parent_node: The parent node being split
        
        Returns:
            Index where to split (exclusive for first half, inclusive for second half)
        """
        # Create a summary of each child topic
        children_summary = []
        for i, child in enumerate(topic_children):
            children_summary.append(f"[{i}] {child.topic_name} - {child.summary[:100]}")
        
        children_text = "\n".join(children_summary)
        
        prompt = f"""You are analyzing a conversation topic "{parent_node.topic_name}" that has been broken down into {len(topic_children)} subtopics.

Your task is to find the BEST SPLIT POINT where there is a natural topic shift or change in discussion focus.

Subtopics (in chronological order):
{children_text}

Analyze the flow of topics and identify where there is a natural break or shift in the conversation theme.

Respond with a JSON object containing:
{{
  "reasoning": "brief explanation of why this is the best split point",
  "split_index": <integer between 1 and {len(topic_children)-1}, representing where to split>
}}

The split_index should be the starting index of the second group. For example:
- split_index=1 means [0] vs [1,2,3,...]
- split_index=3 means [0,1,2] vs [3,4,5,...]

Respond ONLY with valid JSON, no other text."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.2, max_tokens=200)
            result = extract_json(response)
            split_index = int(result.get("split_index", len(topic_children) // 2))
            
            # Validate split_index
            split_index = max(1, min(split_index, len(topic_children) - 1))
            
            return split_index
            
        except Exception as e:
            print(f"LLM error in finding split point: {e}")
            # Fallback: split in the middle
            return len(topic_children) // 2
    
    def _llm_generate_topic_from_children(self, topic_children: List[TopicNode], parent_node: TopicNode) -> str:
        """
        Generate a topic name that encompasses a group of child topics.
        
        Args:
            topic_children: List of TopicNode children
            parent_node: The original parent node
        
        Returns:
            Generated topic name
        """
        # Create a summary of the children
        children_names = [child.topic_name for child in topic_children]
        children_text = ", ".join(children_names[:5])  # Limit to first 5
        if len(children_names) > 5:
            children_text += f", and {len(children_names) - 5} more"
        
        prompt = f"""Generate a concise topic name (2-6 words) that captures the common theme of these subtopics:

Parent topic: {parent_node.topic_name}

Subtopics: {children_text}

The new topic name should:
1. Be more specific than the parent topic
2. Capture the essence of the listed subtopics
3. Be concise (2-6 words)

Respond with ONLY the topic name, nothing else."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.3, max_tokens=50)
            topic_name = response.strip()
            
            # Remove quotes if present
            topic_name = topic_name.strip('"\'')
            
            return topic_name
            
        except Exception as e:
            print(f"LLM error in generating topic from children: {e}")
            # Fallback: use parent name with part number
            return f"{parent_node.topic_name} - Part"
    
    def _llm_group_topics_into_subtopics(self, topic_children: List[TopicNode], parent_node: TopicNode) -> List[Dict[str, Any]]:
        """
        Use LLM to group topic children into logical subtopic groups.
        
        Args:
            topic_children: List of TopicNode children to group
            parent_node: The parent node (typically root)
        
        Returns:
            List of subtopic group dictionaries
        """
        # Create a summary of each child topic
        topics_summary = []
        for i, child in enumerate(topic_children):
            topics_summary.append(f"[{i}] {child.topic_name} - {child.summary[:100]}")
        
        topics_text = "\n".join(topics_summary)
        
        # Determine number of groups (aim for max_children topics per group)
        num_topics = len(topic_children)
        num_groups = max(2, (num_topics + self.max_children - 1) // self.max_children)
        
        prompt = f"""Analyze the following {num_topics} conversation topics and group them into {num_groups} logical categories based on thematic similarity.
        
Topics (in chronological order):
{topics_text}

Your task is to create {num_groups} groups where each group contains related topics. Topics should be grouped by theme/subject matter, but also respect chronological flow when possible.

Respond with a JSON object containing:
{{
  "groups": [
    {{
      "topic_name": "descriptive name for this group of topics (2-6 words)",
      "summary": "brief description of what this group covers",
      "topic_indices": [list of topic indices (0-{num_topics-1}) that belong to this group]
    }},
    ...
  ]
}}

Ensure:
- Exactly {num_groups} groups
- All topics are assigned to exactly one group
- Topics in each group are thematically related
- Groups respect chronological ordering where possible (prefer consecutive indices in each group)

Respond ONLY with valid JSON, no other text."""
        
        try:
            response = ChatGPT_API(self.model, prompt, api_key=self.api_key, temperature=0.3, max_tokens=800)
            result = extract_json(response)
            
            # Handle different response formats
            if isinstance(result, dict) and "groups" in result:
                groups = result["groups"]
            elif isinstance(result, list):
                groups = result
            else:
                # Fallback: create simple sequential groups
                return self._create_fallback_groups(topic_children, num_groups)
            
            # Validate that all topics are covered
            all_indices = set()
            for group in groups:
                all_indices.update(group.get("topic_indices", []))
            
            # If validation fails, use fallback
            if len(all_indices) != num_topics:
                print(f"Warning: LLM grouping didn't cover all topics, using fallback")
                return self._create_fallback_groups(topic_children, num_groups)
            
            return groups
            
        except Exception as e:
            print(f"LLM error in grouping topics: {e}")
            # Fallback: create simple sequential groups
            return self._create_fallback_groups(topic_children, num_groups)
    
    def _create_fallback_groups(self, topic_children: List[TopicNode], num_groups: int) -> List[Dict[str, Any]]:
        """
        Create fallback topic groups when LLM fails.
        
        Args:
            topic_children: List of TopicNode children
            num_groups: Number of groups to create
        
        Returns:
            List of group dictionaries
        """
        groups = []
        topics_per_group = len(topic_children) // num_groups
        
        for i in range(num_groups):
            start_idx = i * topics_per_group
            # Last group gets any remaining topics
            end_idx = start_idx + topics_per_group if i < num_groups - 1 else len(topic_children)
            
            topic_indices = list(range(start_idx, end_idx))
            group_topics = [topic_children[j] for j in topic_indices]
            
            groups.append({
                "topic_name": f"Topic Group {i + 1}",
                "summary": f"Group of topics from {group_topics[0].topic_name} to {group_topics[-1].topic_name}",
                "topic_indices": topic_indices
            })
        
        return groups
    
    def generate_summaries(self) -> None:
        """
        Generate summaries for all frozen nodes (nodes that won't change anymore).
        
        This is a public method that can be called to generate summaries on demand.
        Summaries are automatically generated during normal operation, but this method
        can be useful after loading a tree or for manual summary generation.
        """
        self._generate_summaries_for_frozen_nodes()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export the tree structure to a dictionary."""
        result = {
            "max_children": self.max_children,
            "total_messages": len(self.conversation),
            "tree": self.root.to_dict()
        }
        
        return result
    
    def save(self, filepath: str, save_conversation: bool = False) -> None:
        """
        Save the tree to a JSON file.
        
        This saves CTree state including:
        - max_children configuration
        - tree structure (all nodes)
        - full conversation history (optional)
        
        Args:
            filepath: Path to save the tree JSON file
            save_conversation: If True, includes full conversation history in saved file.
                             If False (default), only saves tree structure.
        """
        # Generate summaries for frozen nodes before saving
        self._generate_summaries_for_frozen_nodes()
        
        data = self.to_dict()
        # Optionally save the full conversation history
        if save_conversation:
            data['conversation'] = self.conversation
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str, api_key: Optional[str] = None, model: Optional[str] = None) -> 'CTree':
        """
        Load a CTree from a JSON file (class method).
        
        This restores all CTree state including:
        - max_children configuration
        - full conversation history
        - tree structure (all nodes)
        - current_node pointer
        
        Args:
            filepath: Path to the saved tree JSON file
            api_key: OpenAI API key (optional if set in environment)
            model: LLM model to use (if None, defaults to 'gpt-4o-mini')
        
        Returns:
            Loaded CTree instance
            
        Example:
            >>> tree = CTree.load('my_tree.json')
            >>> print(f"Loaded {len(tree.conversation)} messages")
            >>> tree.print_tree()
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use default model if not provided
        if model is None:
            model = 'gpt-4o-mini'
        
        # Create new tree instance with saved parameters
        tree = cls(
            max_children=data.get('max_children', 5),
            api_key=api_key,
            model=model
        )
        
        # Restore conversation history
        tree.conversation = data.get('conversation', [])
        
        # Reconstruct the tree structure
        tree.root = tree._reconstruct_node(data['tree'], parent=None)
        
        # Find the current node (rightmost/most recent node)
        tree.current_node = tree._find_current_node(tree.root)
        
        return tree
    
    def _reconstruct_node(self, node_data: Dict[str, Any], parent: Optional[Node]) -> Node:
        """
        Recursively reconstruct nodes from dictionary data.
        
        Args:
            node_data: Dictionary containing node data
            parent: Parent node reference
        
        Returns:
            Reconstructed Node
        """
        node_type = node_data.get('type', 'topic')
        
        if node_type == 'message' :  # Support both for backward compatibility
            # Reconstruct MessageNode
            # Support both message_index (new, position in conversation) and pair_index (old, exchange count)
            if 'message_index' in node_data:
                # New format: message_index is the position in conversation list
                msg_start_idx = node_data['message_index']
            elif 'pair_index' in node_data:
                # Old format: pair_index is the exchange count, convert to position
                msg_start_idx = node_data['pair_index'] * 2
            else:
                msg_start_idx = 0
            
            # Get the actual messages from conversation history
            user_msg = {}
            assistant_msg = {}
            system_msg = None
            
            # Scan conversation starting from the index
            for i in range(max(0, msg_start_idx - 1), min(len(self.conversation), msg_start_idx + 4)):
                if i < len(self.conversation) and self.conversation[i].get('role') == 'user':
                    # Check if previous message is system
                    if i > 0 and self.conversation[i - 1].get('role') == 'system':
                        system_msg = self.conversation[i - 1]
                    user_msg = self.conversation[i]
                    # Next message should be assistant
                    if i + 1 < len(self.conversation) and self.conversation[i + 1].get('role') == 'assistant':
                        assistant_msg = self.conversation[i + 1]
                        break
            
            node = MessageNode(
                user_message=user_msg,
                assistant_message=assistant_msg,
                system_message=system_msg,
                message_index=msg_start_idx,
                parent=parent
            )
            node.sub_node_count = node_data.get('sub_node_count', 0)
            
        else:
            # Reconstruct TopicNode
            node = TopicNode(
                topic_name=node_data.get('topic_name', ''),
                summary=node_data.get('summary', ''),
                start_index=node_data.get('start_index', 0),
                end_index=node_data.get('end_index', 0),
                parent=parent
            )
            node.sub_node_count = node_data.get('sub_node_count', 0)
            
            # Recursively reconstruct children
            for child_data in node_data.get('children', []):
                child_node = self._reconstruct_node(child_data, parent=node)
                node.children.append(child_node)
        
        return node
    
    def _find_current_node(self, node: Node) -> TopicNode:
        """
        Find the current (most recent) node in the tree.
        This is typically the rightmost leaf topic node or the last topic with children.
        
        Args:
            node: Node to start search from
        
        Returns:
            Current TopicNode
        """
        if isinstance(node, MessageNode):
            # MessageNodes can't be current, return parent
            return node.parent if isinstance(node.parent, TopicNode) else self.root
        
        if not isinstance(node, TopicNode):
            return self.root
        
        # If node has children, check the last child
        if node.children:
            last_child = node.children[-1]
            
            # If last child is a MessageNode, current node is this topic
            if isinstance(last_child, MessageNode):
                return node
            
            # If last child is a TopicNode, recurse into it
            if isinstance(last_child, TopicNode):
                return self._find_current_node(last_child)
        
        # If no children, this is the current node
        return node
    
    def print_tree(self, node: Optional[Node] = None, indent: int = 0, show_messages: bool = False) -> None:
        """
        Print a visual representation of the tree.
        
        Args:
            node: Node to start printing from (defaults to root)
            indent: Current indentation level
            show_messages: If True, shows individual message nodes (can be verbose)
        """
        if node is None:
            node = self.root
        
        prefix = "  " * indent
        
        if isinstance(node, TopicNode):
            msg_count = node.get_message_count()
            
            if node is self.root:
                print(f"{prefix}ROOT (sub-nodes: {node.sub_node_count})")
            else:
                print(f"{prefix}├─ {node.topic_name} [{node.start_index}:{node.end_index}] ({msg_count} msgs, {node.sub_node_count} sub-nodes)")
                if indent < 3:  # Only show summary for first few levels
                    print(f"{prefix}   {node.summary[:100]}")
        
        elif isinstance(node, MessageNode):
            if show_messages:  # Only show messages if explicitly requested
                if node.system_message:
                    system_content = node.system_message.get("content", "")[:50]
                    print(f"{prefix}  └─ Message[{node.message_index}] (3 msgs):")
                    print(f"{prefix}     System: {system_content}...")
                else:
                    print(f"{prefix}  └─ Message[{node.message_index}] (2 msgs):")
                user_content = node.user_message.get("content", "")[:50]
                assistant_content = node.assistant_message.get("content", "")[:50]
                print(f"{prefix}     User: {user_content}...")
                print(f"{prefix}     Assistant: {assistant_content}...")
        
        # Recursively print children
        for child in node.children:
            # For topic nodes, always recurse
            # For message nodes, only if show_messages is True
            if isinstance(child, TopicNode) or (isinstance(child, MessageNode) and show_messages):
                self.print_tree(child, indent + 1, show_messages)



