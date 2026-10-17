import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

target_id = 'Duke_person2250'
queries = [
    ('Exact Dataset Description 1', 'The man is wearing a white t-shirt and he is wearing blue pants. He has short and black hair. He is wearing a white shoes.'),
    ('Exact Dataset Description 2', 'The man, who has short black hair, is wearing a white t-shirt and blue pants. He also has white shoes on.'),
    ('Natural Rephrasing', 'The man is wearing a white t-shirt with blue trousers and white sneakers.'),
    ('Short Witness Query', 'Man in white shirt and blue pants with short hair.')
]

print(f"Testing Query Variations for Target {target_id} across 820 Videos:")
print("=" * 75)

for label, q in queries:
    payload = json.dumps({
        'query': q,
        'condition': 'corrupted',
        'method': 'weighted',
        'top_k': 820
    }).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:5000/api/search', data=payload, headers={'Content-Type': 'application/json'})
    res = json.loads(urllib.request.urlopen(req).read().decode())
    
    # Find rank of target
    target_rank = None
    target_score = None
    for r, cand in enumerate(res['results'], 1):
        if cand['id'] == 'person2250':
            target_rank = r
            target_score = cand['score']
            break
    
    print(f"\n[{label}]")
    print(f"  Query: \"{q}\"")
    print(f"  Target person2250: Rank #{target_rank} of 820 (Score: +{target_score:.4f})")
    print(f"  Top Candidate: {res['results'][0]['id']} ({res['results'][0]['dataset']}) - Score: +{res['results'][0]['score']:.4f}")
