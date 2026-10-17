import os
import json
import torch
import argparse
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger_eng')

valid_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}

def extract_keywords(text, stop_words):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in stop_words]
    tagged_words = nltk.pos_tag(words)
    keywords = [word for word, tag in tagged_words if tag in valid_pos_tags]
    return keywords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    args = parser.parse_args()
    
    data_dir = f"../data/TVPReID/TVPReid-{args.dataset}"
    json_path = os.path.join(data_dir, 'captions', f'TVPReid-{args.dataset}.json')
    
    with open(json_path, 'r') as f:
        video_to_captions = json.load(f)
        
    stop_words = set(stopwords.words('english'))
    
    print("Extracting keywords to form Vocab Bank...")
    vocab_bank = set()
    video_keywords = {}
    
    for vid, captions in video_to_captions.items():
        combined_text = " ".join(captions)
        keywords = extract_keywords(combined_text, stop_words)
        video_keywords[vid] = keywords
        vocab_bank.update(keywords)
        
    vocab_list = sorted(list(vocab_bank))
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    D_tips = len(vocab_list)
    print(f"Vocab Bank size (D_tips): {D_tips}")
    
    print("Generating frequency-weighted Tips for each video...")
    tips_dict = {}
    for vid, keywords in video_keywords.items():
        tip_vector = torch.zeros(D_tips)
        if not keywords:
            tips_dict[vid] = tip_vector
            continue
            
        counter = Counter(keywords)
        total_keywords = len(keywords) # C in the paper
        
        for word, count in counter.items():
            idx = word_to_idx[word]
            weight = count / total_keywords # w_n = c_n / C
            tip_vector[idx] = weight
            
        tips_dict[vid] = tip_vector
        
    # Save outputs
    out_dir = "data/"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, f'{args.dataset}_vocab_bank.json'), 'w') as f:
        json.dump(vocab_list, f)
        
    torch.save(tips_dict, os.path.join(out_dir, f'{args.dataset}_tips.pt'))
    print(f"Saved {args.dataset}_vocab_bank.json and {args.dataset}_tips.pt successfully!")

if __name__ == "__main__":
    main()
