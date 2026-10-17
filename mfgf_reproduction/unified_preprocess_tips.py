import os
import json
import torch
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')

VALID_POS_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}

def extract_keywords(text, stop_words):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in stop_words]
    tagged_words = nltk.pos_tag(words)
    keywords = [word for word, tag in tagged_words if tag in VALID_POS_TAGS]
    return keywords

def main():
    print("=" * 60)
    print("Generating Unified Vocabulary Bank and TIPS Vectors")
    print("Datasets: PRID, iLIDs, Duke")
    print("=" * 60)

    datasets = ['PRID', 'iLIDs', 'Duke']
    stop_words = set(stopwords.words('english'))
    
    all_video_keywords = {}
    vocab_bank = set()
    
    for ds in datasets:
        json_path = f"../data/TVPReID/TVPReid-{ds}/captions/TVPReid-{ds}.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Captions file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            video_to_captions = json.load(f)
            
        print(f"Loaded {len(video_to_captions)} captions from {ds}")
        
        for vid, captions in video_to_captions.items():
            clean_vid = vid.replace('.mp4', '')
            unified_key = f"{ds}_{clean_vid}"
            combined_text = " ".join(captions)
            keywords = extract_keywords(combined_text, stop_words)
            all_video_keywords[unified_key] = keywords
            vocab_bank.update(keywords)
            
    vocab_list = sorted(list(vocab_bank))
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    D_tips = len(vocab_list)
    print(f"\nUnified Vocab Bank size (D_tips): {D_tips} unique keywords")
    print(f"Total videos processed: {len(all_video_keywords)}")
    
    print("Generating frequency-weighted Tips vectors for unified videos...")
    tips_dict = {}
    for vid, keywords in all_video_keywords.items():
        tip_vector = torch.zeros(D_tips)
        if not keywords:
            tips_dict[vid] = tip_vector
            continue
            
        counter = Counter(keywords)
        total_keywords = len(keywords)
        for word, count in counter.items():
            idx = word_to_idx[word]
            weight = count / total_keywords
            tip_vector[idx] = weight
            
        tips_dict[vid] = tip_vector
        
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    
    vocab_save_path = os.path.join(out_dir, "Unified_vocab_bank.json")
    tips_save_path = os.path.join(out_dir, "Unified_tips.pt")
    
    with open(vocab_save_path, 'w') as f:
        json.dump(vocab_list, f)
        
    torch.save(tips_dict, tips_save_path)
    print(f"\nSaved {vocab_save_path} and {tips_save_path} successfully!")

if __name__ == "__main__":
    main()
