import json, re

SYNONYMS = {
    'coat': 'jacket',
    'jacket': 'jacket',
    'pants': 'trousers',
    'trousers': 'trousers',
    'jeans': 'trousers',
    'bag': 'backpack',
    'handbag': 'backpack',
    'backpack': 'backpack',
    'top': 'shirt',
    'shirt': 'shirt',
    'tshirt': 'shirt',
    'tee': 'shirt',
    'sneakers': 'shoes',
    'boots': 'shoes'
}

STOPWORDS = {'the', 'is', 'a', 'an', 'and', 'wearing', 'with', 'has', 'of', 'pair', 'she', 'he', 'was', 'in', 'on', 'her', 'his', 'walking', 'walks', 'person', 'someone'}
COLORS = {'red', 'blue', 'green', 'black', 'white', 'yellow', 'grey', 'gray', 'brown', 'purple', 'pink', 'orange', 'dark', 'light'}
CLOTHING = {'jacket', 'trousers', 'shirt', 'backpack', 'shoes', 'skirt', 'dress', 'glasses', 'shorts'}

def normalize_tokens(text):
    words = re.findall(r'\w+', text.lower())
    return [SYNONYMS.get(w, w) for w in words if w not in STOPWORDS]

def score_attributes(query, caption):
    q_tokens = normalize_tokens(query)
    c_tokens = normalize_tokens(caption)
    if not q_tokens or not c_tokens:
        return 0.0
    
    score = 0.0
    max_possible = 0.0
    for w in set(q_tokens):
        weight = 5.0 if w in COLORS else (3.0 if w in CLOTHING else 1.0)
        max_possible += weight
        if w in c_tokens:
            score += weight
        else:
            if w in COLORS:
                score -= 3.0 # strict penalty for wrong color!
                
    sim = max(0.0, score / max_possible)
    return sim

with open('demo_data.json') as f:
    d = json.load(f)

q = 'The woman is wearing a red jacket and black trousers'
gallery = d['sample_gallery']
scored = []
for item in gallery:
    max_s = 0.0
    best_c = ''
    for cap in item.get('captions', []):
        s = score_attributes(q, cap)
        if s > max_s:
            max_s = s
            best_c = cap
    scored.append((max_s, item['id'], item['dataset'], best_c))

scored.sort(key=lambda x: x[0], reverse=True)
print('Top 5 matches for query:', q)
for rank, (sim, vid, ds, cap) in enumerate(scored[:5], 1):
    print(f'#{rank}: {ds} {vid} (score: {sim:.3f}) -> "{cap[:75]}..."')
