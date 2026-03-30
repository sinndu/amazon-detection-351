import gzip
import json

FILE_PATH = './data/raw/Electronics.jsonl.gz'

def preview_dataset(file_path, num_reviews=10):
    print(f"Previewing first {num_reviews} reviews from {file_path}:")
    
    with gzip.open(file_path, mode='rt', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= num_reviews:
                break
            review_data = json.loads(line)

            print(f"\nReview #{count + 1}:")
            print(json.dumps(review_data, indent=4))
            
            count += 1
if __name__ == "__main__":
    preview_dataset(FILE_PATH)

    