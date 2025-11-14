"""
Visualization utilities for CTree.

This module provides functions to visualize the conversation tree structure
in various formats including text-based trees and JSON exports.
"""

import json
from typing import Optional, List, Dict, Any
from .ctree import CTree, TopicNode


def print_detailed_tree(tree: CTree, max_depth: Optional[int] = None) -> None:
    """
    Print a detailed tree structure with statistics.
    
    Args:
        tree: The CTree to visualize
        max_depth: Maximum depth to display (None for unlimited)
    """
    print("="*80)
    print("CONVERSATION TREE - DETAILED VIEW")
    print("="*80)
    print(f"Total messages: {len(tree.conversation)}")
    print(f"Max children: {tree.max_children}")
    print("="*80)
    print()
    
    def print_node(node: TopicNode, depth: int = 0, is_last: bool = True, prefix: str = ""):
        # Skip root in display
        if node.topic_name == "ROOT":
            for i, child in enumerate(node.children):
                print_node(child, 0, i == len(node.children) - 1, "")
            return
        
        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            return
        
        # Prepare node info
        msg_count = node.get_message_count()
        child_count = len(node.children)
        
        # Create tree connectors
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        # Print node
        print(f"{prefix}{connector}{node.topic_name}")
        print(f"{prefix}{extension}├─ Range: [{node.start_index}:{node.end_index}]")
        print(f"{prefix}{extension}├─ Messages: {msg_count}")
        print(f"{prefix}{extension}├─ Children: {child_count}")
        
        # Print summary (truncated)
        summary = node.summary.replace('\n', ' ')[:100]
        if len(node.summary) > 100:
            summary += "..."
        print(f"{prefix}{extension}└─ Summary: {summary}")
        
        # Print children
        if child_count > 0:
            print(f"{prefix}{extension}")
            new_prefix = prefix + extension
            for i, child in enumerate(node.children):
                print_node(child, depth + 1, i == child_count - 1, new_prefix)
    
    print_node(tree.root)


def export_tree_statistics(tree: CTree) -> Dict[str, Any]:
    """
    Generate statistics about the tree structure.
    
    Args:
        tree: The CTree to analyze
    
    Returns:
        Dictionary containing various statistics
    """
    stats = {
        "total_messages": len(tree.conversation),
        "max_children": tree.max_children,
        "total_nodes": 0,
        "leaf_nodes": 0,
        "max_depth": 0,
        "avg_messages_per_leaf": 0,
        "topic_distribution": []
    }
    
    leaf_message_counts = []
    
    def analyze_node(node: TopicNode, depth: int = 0):
        if node.topic_name == "ROOT":
            for child in node.children:
                analyze_node(child, 0)
            return
        
        stats["total_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        
        if len(node.children) == 0:
            stats["leaf_nodes"] += 1
            leaf_message_counts.append(node.get_message_count())
            stats["topic_distribution"].append({
                "topic": node.topic_name,
                "depth": depth,
                "messages": node.get_message_count(),
                "range": [node.start_index, node.end_index]
            })
        
        for child in node.children:
            analyze_node(child, depth + 1)
    
    analyze_node(tree.root)
    
    if leaf_message_counts:
        stats["avg_messages_per_leaf"] = sum(leaf_message_counts) / len(leaf_message_counts)
    
    return stats


def print_statistics(tree: CTree) -> None:
    """
    Print comprehensive statistics about the tree.
    
    Args:
        tree: The CTree to analyze
    """
    stats = export_tree_statistics(tree)
    
    print("="*80)
    print("TREE STATISTICS")
    print("="*80)
    print(f"Total messages:           {stats['total_messages']}")
    print(f"Max children:             {stats['max_children']}")
    print(f"Total topic nodes:        {stats['total_nodes']}")
    print(f"Leaf nodes:               {stats['leaf_nodes']}")
    print(f"Maximum depth:            {stats['max_depth']}")
    print(f"Avg messages per leaf:    {stats['avg_messages_per_leaf']:.2f}")
    print("="*80)
    
    if stats['topic_distribution']:
        print("\nLEAF TOPIC DISTRIBUTION:")
        print("-"*80)
        for topic_info in stats['topic_distribution'][:10]:  # Show first 10
            indent = "  " * topic_info['depth']
            print(f"{indent}{topic_info['topic']}: {topic_info['messages']} messages "
                  f"[{topic_info['range'][0]}:{topic_info['range'][1]}]")
        
        if len(stats['topic_distribution']) > 10:
            print(f"... and {len(stats['topic_distribution']) - 10} more topics")


def export_tree_to_markdown(tree: CTree, filepath: str) -> None:
    """
    Export the tree structure to a Markdown file.
    
    Args:
        tree: The CTree to export
        filepath: Path to save the Markdown file
    """
    lines = []
    
    lines.append("# Conversation Tree Structure\n")
    lines.append(f"**Total Messages:** {len(tree.conversation)}\n")
    lines.append(f"**Max Children:** {tree.max_children}\n")
    lines.append("\n---\n")
    
    def format_node(node: TopicNode, depth: int = 0):
        if node.topic_name == "ROOT":
            for child in node.children:
                format_node(child, 0)
            return
        
        indent = "  " * depth
        msg_count = node.get_message_count()
        
        # Create markdown header
        header_level = min(depth + 2, 6)  # Max header level is 6
        lines.append(f"\n{'#' * header_level} {node.topic_name}\n")
        lines.append(f"\n**Range:** `[{node.start_index}:{node.end_index}]` "
                    f"({msg_count} messages)\n")
        lines.append(f"\n**Summary:** {node.summary}\n")
        
        if node.children:
            lines.append(f"\n**Subtopics:** {len(node.children)}\n")
        
        for child in node.children:
            format_node(child, depth + 1)
    
    format_node(tree.root)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Tree structure exported to {filepath}")


def export_coverage_report(tree: CTree, filepath: str) -> None:
    """
    Export a coverage report showing which messages are in which topics.
    
    Args:
        tree: The CTree to analyze
        filepath: Path to save the report
    """
    coverage = []
    
    def collect_coverage(node: TopicNode, path: List[str] = []):
        if node.topic_name == "ROOT":
            for child in node.children:
                collect_coverage(child, [])
            return
        
        current_path = path + [node.topic_name]
        
        # If leaf node, record coverage
        if len(node.children) == 0:
            for i in range(node.start_index, node.end_index):
                coverage.append({
                    "message_index": i,
                    "topic_path": " > ".join(current_path),
                    "topic_depth": len(current_path)
                })
        
        # Recurse to children
        for child in node.children:
            collect_coverage(child, current_path)
    
    collect_coverage(tree.root)
    
    # Sort by message index
    coverage.sort(key=lambda x: x["message_index"])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    
    print(f"Coverage report exported to {filepath}")


def visualize_tree_ascii(tree: CTree, max_width: int = 80) -> str:
    """
    Create an ASCII art visualization of the tree.
    
    Args:
        tree: The CTree to visualize
        max_width: Maximum width of the visualization
    
    Returns:
        String containing ASCII art representation
    """
    lines = []
    
    def add_node(node: TopicNode, prefix: str = "", is_last: bool = True):
        if node.topic_name == "ROOT":
            for i, child in enumerate(node.children):
                add_node(child, "", i == len(node.children) - 1)
            return
        
        # Truncate topic name if needed
        max_name_len = max_width - len(prefix) - 10
        topic_display = node.topic_name
        if len(topic_display) > max_name_len:
            topic_display = topic_display[:max_name_len-3] + "..."
        
        # Add connector
        connector = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{connector}{topic_display} [{node.start_index}:{node.end_index}]")
        
        # Prepare prefix for children
        if node.children:
            new_prefix = prefix + ("   " if is_last else "│  ")
            for i, child in enumerate(node.children):
                add_node(child, new_prefix, i == len(node.children) - 1)
    
    add_node(tree.root)
    return "\n".join(lines)


def main():
    """Main function for visualization examples."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <tree_json_file> [command]")
        print("\nCommands:")
        print("  detailed   - Show detailed tree view")
        print("  stats      - Show statistics")
        print("  markdown   - Export to markdown (saves as tree.md)")
        print("  coverage   - Export coverage report (saves as coverage.json)")
        print("  ascii      - Show ASCII art tree")
        return
    
    # Load tree from JSON
    tree_file = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "detailed"
    
    print(f"Loading tree from {tree_file}...")
    
    # Note: This loads the saved tree structure, not a full CTree instance
    # For full functionality, you'd need to reconstruct the tree
    with open(tree_file, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    print(f"Tree has {tree_data['total_messages']} messages")
    print(f"Max children: {tree_data.get('max_children', 'N/A')}")
    print("\nFor full visualization, reconstruct the tree with CTree")


if __name__ == "__main__":
    main()






