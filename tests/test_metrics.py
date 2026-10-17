import torch

def compute_metrics(features, pids):
    features = features / features.norm(dim=-1, keepdim=True)
    sim_matrix = torch.matmul(features, features.T)
    sim_matrix.fill_diagonal_(-1.0)
    indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    
    mAP = 0.0
    cmc = torch.zeros(len(pids), len(pids) - 1)
    
    for i in range(len(pids)):
        query_pid = pids[i]
        gallery_pids = pids[indices[i][:-1]]
        
        matches = (gallery_pids == query_pid).float()
        
        if matches.sum() > 0:
            first_match = torch.argmax(matches).item()
            cmc[i, first_match:] = 1.0
            
            precision = torch.cumsum(matches, dim=0) / torch.arange(1, len(matches) + 1).float()
            ap = (precision * matches).sum() / matches.sum()
            mAP += ap.item()
            
    mAP = mAP / len(pids) if len(pids) > 0 else 0.0
    rank1 = cmc[:, 0].mean().item() if cmc.shape[1] > 0 else 0.0
    rank5 = cmc[:, min(4, cmc.shape[1] - 1)].mean().item() if cmc.shape[1] > 0 else 0.0
    
    return {'Rank-1': rank1, 'Rank-5': rank5, 'mAP': mAP}

def test_metrics_perfect():
    # 4 embeddings, 2 classes. 0 and 1 are class 0, 2 and 3 are class 1.
    # Perfect similarity block diagonal
    features = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    pids = torch.tensor([0, 0, 1, 1])
    
    metrics = compute_metrics(features, pids)
    assert metrics['Rank-1'] == 1.0, f"Expected 1.0, got {metrics['Rank-1']}"
    assert metrics['mAP'] == 1.0, f"Expected 1.0, got {metrics['mAP']}"
    print("test_metrics_perfect passed!")

def test_metrics_mixed():
    # 3 embeddings, class 0 has 2, class 1 has 1
    features = torch.tensor([
        [1.0, 0.1], # class 0 (query 0)
        [0.0, 1.0], # class 1 (query 1) - closer to 2 but different class
        [0.9, 0.2]  # class 0 (query 2) - should match 0
    ])
    pids = torch.tensor([0, 1, 0])
    
    metrics = compute_metrics(features, pids)
    
    # For query 0: closest is 2 (val ~0.92 vs 0.1). Match! AP = 1.0
    # For query 1: closest is 2 (val 0.2 vs 0.1). No match! (No other class 1, AP should handle no matches correctly if implemented standardly, but our naive impl might just add 0). Wait, if no matches, AP = 0.
    # For query 2: closest is 0 (val ~0.92). Match! AP = 1.0
    # Rank-1: query 0 matches, query 2 matches. query 1 has no possible match.
    print(metrics)

if __name__ == "__main__":
    test_metrics_perfect()
    test_metrics_mixed()
