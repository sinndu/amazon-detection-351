import os
import time
from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models_to_test = [
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash"
]

def test_models():
    print(f"{'MODEL NAME':<25} | {'STATUS':<15} | {'DETAILS'}")
    print("-" * 70)
    
    for model_id in models_to_test:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents="hi"
            )
            print(f"{model_id:<25} | SUCCESS         | Response: {response.text.strip()[:20]}...")
            
        except exceptions.ResourceExhausted:
            print(f"{model_id:<25} | QUOTA HIT       | Limit is 0 or RPM limit.")
        except exceptions.NotFound:
            print(f"{model_id:<25} | NOT FOUND       | Deprecated or typo.")
        except exceptions.PermissionDenied:
            print(f"{model_id:<25} | NO ACCESS       | Key permissions issue.")
        except Exception as e:
            error_msg = str(e)
            if "limit: 0" in error_msg:
                print(f"{model_id:<25} | LIMIT 0         | API enforcing zero quota.")
            else:
                print(f"{model_id:<25} | ERROR           | {error_msg[:50]}")
        
        time.sleep(1)

if __name__ == "__main__":
    test_models()