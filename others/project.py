import os
import shutil
import zipfile
import pandas as pd
import requests
import json
import logging
from fastapi import FastAPI, File, Form, UploadFile
from dotenv import load_dotenv
from typing import Optional

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set in the environment variables.")

app = FastAPI()

# Directory for saving uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_file(file_path: str) -> str:
    """
    Process different file types and extract relevant content.
    """
    ext = file_path.split(".")[-1].lower()

    # CSV files: Extract the first few rows as text
    if ext == "csv":
        df = pd.read_csv(file_path)
        return df.head().to_string()

    # Text-based files: Read and return the first 5000 characters
    elif ext in ["md", "txt", "html", "xml", "json"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()[:5000]  # Limit size to prevent overloading LLM

    # ZIP files: Extract and process contents
    elif ext == "zip":
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(UPLOAD_DIR)
            extracted_files = os.listdir(UPLOAD_DIR)

            # Identify supported file formats inside ZIP
            supported_files = [f for f in extracted_files if f.endswith((".csv", ".md", ".txt", ".json"))]

            if supported_files:
                extracted_file_path = os.path.join(UPLOAD_DIR, supported_files[0])
                return process_file(extracted_file_path)
            else:
                return f"ZIP extracted but no readable files found: {extracted_files}"
        except Exception as e:
            return f"Error extracting ZIP file: {str(e)}"

    # Database files: Indicate that processing is required
    elif ext in ["db", "sqlite"]:
        return f"Database file {file_path} uploaded. Further processing needed."

    # Unknown file types
    return f"File format '{ext}' is not directly supported for extraction."

@app.post("/api/")
async def answer_question(
    question: str = Form(...), 
    file: Optional[UploadFile] = File(None)  # Make file optional
):
    """
    Accepts a question and an optional file, processes it, and returns an answer.
    """
    file_content = "No file provided."

    # If a file is uploaded, process it
    if file:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract relevant content
        file_content = process_file(file_path)

    # Prepare LLM prompt (question only if no file, otherwise question + file content)
    prompt = f"Question: {question}"
    if file:
        prompt += f"\n\nFile Content:\n{file_content}"

    # Send request to GPT-4o mini via AIPROXY
    try:
        response = requests.post(
            AIPROXY_URL,
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={
                "model": "gpt-4o-mini",  # Use GPT-4o mini model
                "messages": [
                    {"role": "system", "content": "You are an AI assistant. Your only job is to provide direct answers with no explanations. Do not give steps, guides, or instructions. Just return the final answer in a single short sentence."},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        # Log full LLM response for debugging
        response_json = response.json()
        logging.info(f"Full LLM Response: {json.dumps(response_json, indent=2)}")

        if "choices" in response_json and response_json["choices"]:
            return {"answer": response_json["choices"][0]["message"]["content"].strip()}
        else:
            return {"answer": f"Error: Unexpected LLM response: {response_json}"}
    
    except Exception as e:
        return {"answer": f"Error processing the request: {str(e)}"}
