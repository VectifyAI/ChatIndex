import json
from ctree import CTree

print("Loading conversation data...")
with open('data/ChatExample.json', 'r', encoding='utf-8') as f:
    conversation = json.load(f)

print(f"Loaded {len(conversation)} messages")

# Build tree from conversation
print("\nBuilding tree from conversation...")
tree = CTree(max_children=10, auto_save_path='save/conversation_tree.json')


messages = []  

for i, msg in enumerate(conversation):    
    if msg.get("role") == "system":
        messages.append({
            "role": "system",
            "content": msg.get("content")
        })
    elif msg.get("role") == "user":
        messages.append({
            "role": "user",
            "content": msg.get("content")
        })
    elif msg.get("role") == "assistant":
        messages.append({
            "role": "assistant",
            "content": msg.get("content")
        })
        tree.add(messages)
        tree.print_tree()
        messages = []  
