from google import genai
from google.genai import types
import json
from config import gemini_api
import matplotlib.pyplot as plt
import numpy as np

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=gemini_api)


def generate_synthetic_data(num_samples):
    prompt = f"Generate {num_samples} data points similar to spiral data with class labels 0, 1, and 2.  Return as comma-separated values (x1, x2, label)."
    response = client.models.generate_content(
    model="gemini-2.5-flash", contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            ),
        )

    synthetic_data = []
    for line in response.text.strip().split('\n'):
        line = line.strip()
        if not line:  # skip empty lines
            continue
        parts = line.split(',')
        if len(parts) != 3:  # skip malformed lines
            continue
        try:
            synthetic_data.append([float(p.strip()) for p in parts])
        except ValueError:
            # skip lines where conversion fails
            continue

    synthetic_data = np.array(synthetic_data, dtype=float)
    
    print(synthetic_data)
    return synthetic_data

# Example usage
synthetic_data = generate_synthetic_data(100)
X,y = synthetic_data[:,:2], synthetic_data[:,2]
# print(synthetic_data)
plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
plt.show()

