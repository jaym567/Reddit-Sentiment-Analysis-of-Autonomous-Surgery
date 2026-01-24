import json
import csv
from typing import Dict, Set, List

def load_unfiltered(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_filtered(path: str) -> Set[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        ids = set()
        for post_entry in data:
            for item in post_entry['items']:
                ids.add(item['id'])
        return ids

def get_all_unfiltered_items(posts: List[Dict]) -> Dict[str, Dict]:
    items = {}
    
    def process_node(node: Dict, type_str: str):
        items[node['id']] = {
            'id': node['id'],
            'type': type_str,
            'body': node.get('body', node.get('title', '')),
            'author': node.get('author', 'OP')
        }
        for reply in node.get('replies', []):
            process_node(reply, 'comment')
            
    for post in posts:
        process_node(post, 'post')
        for comment in post.get('comments', []):
            process_node(comment, 'comment')
            
    return items

def compare_filters(unfiltered_path: str, filtered_path: str, output_path: str):
    unfiltered_data = load_unfiltered(unfiltered_path)
    filtered_ids = load_filtered(filtered_path)
    
    all_items = get_all_unfiltered_items(unfiltered_data)
    
    rejected_items = []
    for item_id, item_data in all_items.items():
        if item_id not in filtered_ids:
            rejected_items.append(item_data)
            
    print(f"Total items in original: {len(all_items)}")
    print(f"Total items in filtered: {len(filtered_ids)}")
    print(f"Total items rejected: {len(rejected_items)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_original": len(all_items),
                "total_filtered": len(filtered_ids),
                "total_rejected": len(rejected_items)
            },
            "rejected_items": rejected_items
        }, f, indent=2, ensure_ascii=False)
    
    # Export to CSV
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'type', 'author', 'body'])
        writer.writeheader()
        for item in rejected_items:
            writer.writerow(item)
    
    print(f"Audit files saved to:\n  - {output_path}\n  - {csv_path}")
    
    # Print a few examples of rejected items
    print("\nSample Rejected Items:")
    for item in rejected_items[:5]:
        preview = item['body'][:75] + "..." if len(item['body']) > 75 else item['body']
        print(f"- [{item['type'].upper()}] {item['author']}: {preview}")

if __name__ == "__main__":
    unfiltered = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\reddit_robotic_surgery_temporal_flat.json"
    filtered = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\filtered_reddit_robotic_surgery.json"
    output = r"c:\Users\jaymo\OneDrive\Desktop\ARGOS\Sentiment Analysis of Autonomous Surgery\SentimentCode\rejected_items_audit.json"
    
    try:
        compare_filters(unfiltered, filtered, output)
    except PermissionError:
        print("⚠️ Permission denied for audit CSV. It might be open in another program.")
        # Try local fallback
        output_local = "rejected_items_audit_local.json"
        compare_filters(unfiltered, filtered, output_local)
