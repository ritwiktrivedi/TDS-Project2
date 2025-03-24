import os
import requests
from dotenv import load_dotenv

# Load the environment variables from .env file (if you have it)
load_dotenv()

# Get the AIPROXY_TOKEN from environment variable
aiproxy_token = os.getenv("AIPROXY_TOKEN")

# If you have the token as a string, you can also set it directly:
# aiproxy_token = "your_aiproxy_token_here"

# Define the headers and data for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {aiproxy_token}"
}

data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "List only the valid English words from these: k3ADrvzier, VA7p8, AFnT, BKp7, 7CzrQI8L3, cg2, qHXVa4, 1TMgV, 7Q, 7z, 23ZKgSKuE1, n, D, q0S, 9F6Ht, 1P4s, ieK, 2laK9miOr, 9yvQ3, AL0iIKk5UR, VcAMGZkC, 14qRSZ0Jlm, Qt, TmCSbnaOi, GHvIz34qp, S, nfyU8, UD9qc, hv, ZDu0Anl, e, Y, PU, aF, t0W, fCmSl1PObS, EXk, VHcfIyUv, efD1bujZB9, pdAPN6IzNA, W06Xim0Kj, KDaBjaAd, lBORNzf, IzjxPpr, JV, A, uQBWWzi, PrTQi, m, b6zON, w6CDI, Um7Wt, Ues2RDsrO, rA, Ef, Fs, J2Nsqso, bdydMBsIm, 9C5ZO187, G3JHvk, AtHM, rKBKaxl7"}
    ]
}

# Send the POST request to the proxy endpoint
url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
response = requests.post(url, json=data, headers=headers)

# Print the response from the API
if response.status_code == 200:
    response_json = response.json()
    
    # Extracting token usage
    prompt_tokens = response_json['usage']['prompt_tokens']
    completion_tokens = response_json['usage']['completion_tokens']
    total_tokens = response_json['usage']['total_tokens']
    
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    
    # Print the assistant's response
    print(response_json['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")
