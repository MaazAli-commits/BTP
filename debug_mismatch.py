import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_query(q, label):
    for cond, method in [('clean', 'mean'), ('clean', 'weighted'), ('corrupted', 'mean'), ('corrupted', 'weighted')]:
        payload = json.dumps({
            'preset_id': 'preset_1',
            'query': q,
            'condition': cond,
            'method': method,
            'top_k': 6
        }).encode('utf-8')
        r = urllib.request.Request('http://127.0.0.1:5000/api/search', data=payload, headers={'Content-Type': 'application/json'})
        resp = json.loads(urllib.request.urlopen(r).read().decode())
        tinfo = resp.get('target_info', {})
        print(f"  {label} [{cond} - {method}]: Rank #{tinfo.get('rank')} (Score: {tinfo.get('score')})")

print("--- Test with WOMAN query (the bug from screenshot) ---")
test_query("The woman is wearing a white jacket and blue trousers with a red backpack walking.", "WOMAN query")

print("\n--- Test with TRUE PRESET 1 MAN query ---")
test_query("The man is wearing a white t-shirt and he is wearing blue pants. He has short and black hair. He is wearing a white shoes.", "MAN query")
