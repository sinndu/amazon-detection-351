import pandas as pd
import numpy as np
import re

def calculate_text_metrics(text):
    if not isinstance(text, str):
        return 0, 0, 0
    
    words = text.split()
    word_count = len(words)
    
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences) if len(sentences) > 0 else 1 
    
    char_count = len(text)
    
    return word_count, sentence_count, char_count

def generate_dataset_summary(human_file, ai_file):
    column_names = ['text', 'title', 'rating', 'timestamp', 'user_id', 'asin', 'flag']
    
    print("Loading datasets...")
    try:
        df_human = pd.read_csv(human_file, header=None, names=column_names)
        df_human['label'] = 'Human-Written'
        
        df_ai = pd.read_csv(ai_file, header=None, names=column_names)
        df_ai['label'] = 'LLM-Generated'
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    df = pd.concat([df_human, df_ai], ignore_index=True) # merge datasets

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    print("Calculating word and sentence counts...")
    df[['word_count', 'sentence_count', 'char_count']] = df.apply(
        lambda row: pd.Series(calculate_text_metrics(row['text'])), axis=1
    )
    
    print("\nDATASET SUMMARY FOR TABLE 1")
    
    classes = ['Human-Written', 'LLM-Generated']
    summary_data = []
    
    for cls in classes:
        subset = df[df['label'] == cls]
        
        metrics = {
            'Class Label': cls,
            'Total Samples': len(subset),
            'Average Word Count': round(subset['word_count'].mean(), 1),
            'Average Sentence Count': round(subset['sentence_count'].mean(), 1),
            'Mean Rating': round(subset['rating'].mean(), 1)
        }
        summary_data.append(metrics)
        
    summary_df = pd.DataFrame(summary_data)
    summary_table = summary_df.set_index('Class Label').T
    
    print("\n")
    print(summary_table.to_markdown())
    print("\n")

if __name__ == "__main__":
    HUMAN_CSV_PATH = "data\collected\collected-llama-3.1-8b-instant-20260213-1940\human_reviews.csv"
    AI_CSV_PATH = "data\collected\collected-llama-3.1-8b-instant-20260213-1940\synthetic_reviews.csv"
    
    generate_dataset_summary(HUMAN_CSV_PATH, AI_CSV_PATH)