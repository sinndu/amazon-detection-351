import os
import gzip
import json
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai
from datetime import datetime
from groq import Groq
import re

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY") # Grab API key
client = Groq(api_key=API_KEY)

def load_asin_metadata(meta_gz_path): 
    asin_map = {}
    print(f"Loading metadata from {meta_gz_path}:")
    
    with gzip.open(meta_gz_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building Title Map"):
            try:
                data = json.loads(line)
                asin = data.get('parent_asin')
                title = data.get('title')
                
                if asin and title:
                    asin_map[asin] = title
            except json.JSONDecodeError:
                continue
    return asin_map

def generate_ai_review(product_title, rating, word_count):
    prompt = (
            f"Write a {rating}-star Amazon review for '{product_title}'. "
            f"Target length: ~{word_count} words. "
            "IMPORTANT: Do NOT include a title, do NOT include 'Rating:', and do NOT use any introductory text. "
            "Just give me the review body itself."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a realistic Amazon reviewer. You never include titles or labels in your output."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        text = completion.choices[0].message.content.strip()
        text = re.sub(r'^(Title|Rating|Review|Headline):\s*', '', text, flags=re.IGNORECASE) # Strip any headers from AI
        text = text.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
        return text
        
    except Exception as e:
        print(f"Groq Error: {e}")
        return None

def build_dataset(review_gz, meta_gz, output_folder, target=1000):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    asin_to_title = load_asin_metadata(meta_gz) # Load metadata map
    
    # Find matches
    real_matches = []
    
    print(f"Processing reviews from {review_gz}:")
    with gzip.open(review_gz, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            if len(real_matches) >= target: break
            data = json.loads(line)
            
            asin = data.get('parent_asin')
            product_title = asin_to_title.get(asin)

            if not product_title or not data.get('verified_purchase') or not data.get('text'): # Skip if not verified, no text, or not in asin map
                continue
            
            real_entry = { # Save human entry
                'text': data.get('text'),
                'product_title': product_title,
                'rating': data.get('rating'),
                'timestamp': data.get('timestamp'),
                'user_id': data.get('user_id'),
                'asin': asin,
                'label': 0 
            }
            real_matches.append(real_entry)

    print(f"{len(real_matches)} matches found!")

    # Start generating AI reviews
    real_dataset = []
    synthetic_dataset = []
    
    print(f"Generating AI reviews for {len(real_matches)} matches:")
    for real_entry in tqdm(real_matches):
        target_len = len(str(real_entry['text']).split()) # Get word count 
        ai_text = generate_ai_review(real_entry['product_title'], real_entry['rating'], target_len) # Generate synthetic review
        
        if ai_text:
            real_dataset.append(real_entry)
            synthetic_dataset.append({
                'text': ai_text, 
                'product_title': real_entry['product_title'],
                'rating': real_entry['rating'],
                'timestamp': real_entry['timestamp'],
                'user_id': "AI_GENERATED",
                'asin': real_entry['asin'],
                'label': 1
            })
        time.sleep(2.1) # Sleep to keep within rate limits


    pd.DataFrame(real_dataset).to_csv(output_path / 'human_reviews.csv', index=False)
    pd.DataFrame(synthetic_dataset).to_csv(output_path / 'synthetic_reviews.csv', index=False)
    print(f"Files saved in {output_path}")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    model_name = "llama-3.1-8b-instant"
    folder_name = f"collected-{model_name}-{timestamp}"
    base_output_dir = Path('data/collected')
    final_output_path = base_output_dir / folder_name

    build_dataset(
        'data/raw/Electronics.jsonl.gz', 
        'data/raw/meta_Electronics.jsonl.gz', 
        final_output_path,
        1000
    )