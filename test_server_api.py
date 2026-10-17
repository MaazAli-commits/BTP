import urllib.request
import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test():
    # 1. Presets
    req = urllib.request.urlopen('http://127.0.0.1:5000/api/presets')
    presets = json.loads(req.read().decode())
    print(f"Total Presets Loaded: {len(presets)}")
    for p in presets:
        print(f"  {p['id']}: {p['title']} [{p.get('tag')}] ({p['dataset']})")

    # 2. Test search for Preset 1 (Duke_person2250, clean -> #1, corrupted mean -> #5, corrupted weighted -> #1)
    for cond, method in [('clean', 'mean'), ('corrupted', 'mean'), ('corrupted', 'weighted')]:
        payload = json.dumps({
            'preset_id': 'preset_1',
            'condition': cond,
            'method': method,
            'top_k': 6
        }).encode('utf-8')
        r = urllib.request.Request('http://127.0.0.1:5000/api/search', data=payload, headers={'Content-Type': 'application/json'})
        resp = json.loads(urllib.request.urlopen(r).read().decode())
        tinfo = resp.get('target_info', {})
        print(f"\nPreset 1 [{cond} - {method}]: Target Rank = #{tinfo.get('rank')} (Score: {tinfo.get('score')}) | In Top-6: {tinfo.get('in_top_k')}")
        print("  Top 6 results:")
        for res in resp['results']:
            badge = " [TARGET]" if res['is_target'] else ""
            print(f"    Rank #{res['rank']}: {res['id']} ({res['dataset']}) - CosSim: {res['score']:+.4f}{badge}")

    print("\n--- Testing All 8 Presets in Both Modes ---")
    for i in range(1, 9):
        pid = f"preset_{i}"
        for method in ['mean', 'weighted']:
            payload = json.dumps({
                'preset_id': pid,
                'condition': 'corrupted',
                'method': method,
                'top_k': 6
            }).encode('utf-8')
            r = urllib.request.Request('http://127.0.0.1:5000/api/search', data=payload, headers={'Content-Type': 'application/json'})
            resp = json.loads(urllib.request.urlopen(r).read().decode())
            tinfo = resp.get('target_info', {})
            in_grid = "Inside Top-6 Grid" if tinfo.get('in_top_k') else "Outside Top-6 Grid"
            print(f"  {pid} ({resp['preset_title']}) [{method.upper()}]: Rank #{tinfo.get('rank')} -> {in_grid}")

if __name__ == '__main__':
    test()
