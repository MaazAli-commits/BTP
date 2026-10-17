import os
import torch

def merge_features_dict(file_dict_map):
    """
    file_dict_map: list of tuples (dataset_name, relative_path)
    Returns: merged dictionary with prefixed keys
    """
    merged = {}
    for ds, rel_path in file_dict_map:
        if not os.path.exists(rel_path):
            raise FileNotFoundError(f"Feature file missing: {rel_path}")
        data = torch.load(rel_path, map_location='cpu')
        print(f"Loading {ds} from {rel_path} ({len(data)} items)")
        for k, v in data.items():
            clean_k = k.replace('.mp4', '')
            unified_k = f"{ds}_{clean_k}"
            merged[unified_k] = v
    print(f"--> Total merged items: {len(merged)}")
    return merged

def main():
    print("=" * 60)
    print("Merging Cached MFGF Features for Unified Reproduction")
    print("=" * 60)
    
    datasets = ['PRID', 'iLIDs', 'Duke']
    
    # 1. Base Extracted Features (Train & Test)
    train_dest = "data/Unified_features/train"
    test_dest = "data/Unified_features/test"
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)
    
    print("\n--- 1. Merging Base Train Features ---")
    train_map = [(ds, f"data/{ds}_features/train/extracted_features.pt") for ds in datasets]
    train_features = merge_features_dict(train_map)
    torch.save(train_features, os.path.join(train_dest, "extracted_features.pt"))
    print(f"Saved: {os.path.join(train_dest, 'extracted_features.pt')}")
    
    print("\n--- 2. Merging Base Test Features ---")
    test_map = [(ds, f"data/{ds}_features/test/extracted_features.pt") for ds in datasets]
    test_features = merge_features_dict(test_map)
    torch.save(test_features, os.path.join(test_dest, "extracted_features.pt"))
    print(f"Saved: {os.path.join(test_dest, 'extracted_features.pt')}")
    
    # 2. Ablation Features (Train: mean, weighted)
    ablation_train_dest = "data/Unified_features_ablation/train"
    ablation_test_dest = "data/Unified_features_ablation/test"
    os.makedirs(ablation_train_dest, exist_ok=True)
    os.makedirs(ablation_test_dest, exist_ok=True)
    
    for pool in ['mean', 'weighted']:
        print(f"\n--- 3. Merging Train Ablation ({pool}) ---")
        pool_train_map = [(ds, f"data/{ds}_features_ablation/train/{pool}_visual_features.pt") for ds in datasets]
        merged_pool_train = merge_features_dict(pool_train_map)
        torch.save(merged_pool_train, os.path.join(ablation_train_dest, f"{pool}_visual_features.pt"))
        print(f"Saved: {os.path.join(ablation_train_dest, f'{pool}_visual_features.pt')}")

    # 3. Ablation Features (Test: clean mean, clean weighted)
    for pool in ['mean', 'weighted']:
        print(f"\n--- 4. Merging Clean Test Ablation ({pool}) ---")
        pool_test_map = [(ds, f"data/{ds}_features_ablation/test/{pool}_visual_features.pt") for ds in datasets]
        merged_pool_test = merge_features_dict(pool_test_map)
        torch.save(merged_pool_test, os.path.join(ablation_test_dest, f"{pool}_visual_features.pt"))
        print(f"Saved: {os.path.join(ablation_test_dest, f'{pool}_visual_features.pt')}")

    # 4. Corrupted Test Features (Test: corrupted_mean, corrupted_weighted)
    for pool in ['mean', 'weighted']:
        print(f"\n--- 5. Merging Corrupted Test Features ({pool}) ---")
        corrupted_test_map = [(ds, f"data/{ds}_features_ablation/test/corrupted_{pool}.pt") for ds in datasets]
        merged_corrupted_test = merge_features_dict(corrupted_test_map)
        torch.save(merged_corrupted_test, os.path.join(ablation_test_dest, f"corrupted_{pool}.pt"))
        print(f"Saved: {os.path.join(ablation_test_dest, f'corrupted_{pool}.pt')}")

    print("\nAll unified features prepared and saved successfully!")

if __name__ == "__main__":
    main()
