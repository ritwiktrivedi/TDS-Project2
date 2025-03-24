from fastapi import FastAPI, Form, File, UploadFile
import requests
import zipfile
import json
import sqlite3
import pandas as pd
import datetime
import io
import os
import re
import hashlib
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

app = FastAPI()

# Function 1: Compute SHA256 hash of formatted README.md
def hash_prettier_readme(file_content):
    try:
        temp_file = "temp_README.md"
        formatted_file = "formatted_README.md"
        
        # Save the uploaded file
        with open(temp_file, "wb") as f:
            f.write(file_content)
        
        # Ensure Prettier is installed
        if os.system("npx --version") != 0:
            return "Error: Prettier (npx) is not installed. Please install Node.js and Prettier."
        
        # Run Prettier and capture output
        os.system(f"npx -y prettier@3.4.2 {temp_file} --write")
        
        # Compute SHA256 hash of the formatted file
        with open(temp_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Cleanup
        os.remove(temp_file)
        
        return file_hash
    except Exception as e:
        return f"Error: {str(e)}"

# Function 2: Compute the result of a flexible Google Sheets formula
def compute_google_sheets_formula(rows, cols, start, step, constrain_rows, constrain_cols):
    sequence_matrix = np.arange(start, start + rows * cols * step, step).reshape(rows, cols)
    constrained_matrix = sequence_matrix[:constrain_rows, :constrain_cols]
    return np.sum(constrained_matrix)

# Google Sheets formulas require execution in a real spreadsheet, so we let LLM handle Q4 and Q5.

# Function 3: Count Wednesdays in a given date range
def count_weekdays(start_date, end_date, weekday):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    return sum(1 for d in range((end - start).days + 1) if (start + datetime.timedelta(days=d)).weekday() == weekday)

# Function 4: Extract CSV answer from ZIP file
def extract_csv_answer(file_content, column_name="answer"):
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith('.csv'):
                with zip_ref.open(filename) as csv_file:
                    df = pd.read_csv(csv_file)
                    return str(df[column_name].iloc[0]) if column_name in df.columns else "Column not found."
    return "CSV file not found."

# Function 5: Sort JSON objects by age and name
def sort_json_objects(json_str):
    data = json.loads(json_str)
    sorted_data = sorted(data, key=lambda x: (x['age'], x['name']))
    return json.dumps(sorted_data, separators=(',', ':'))

# Function 6: Replace all occurrences of "IITM" with "IIT Madras" in extracted files and compute hash
def replace_text_and_hash(file_content):
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        extracted_folder = "extracted_files"
        os.makedirs(extracted_folder, exist_ok=True)
        zip_ref.extractall(extracted_folder)
        for root, _, files in os.walk(extracted_folder):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                content = re.sub(r"(?i)iitm", "IIT Madras", content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        hash_value = hashlib.sha256("".join(sorted(os.listdir(extracted_folder))).encode()).hexdigest()
        return hash_value


# Function 8: Calculate total sales for "Gold" ticket type in a SQLite database
def calculate_gold_ticket_sales(file_content):
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith('.db'):
                with zip_ref.open(filename) as db_file:
                    with open("tickets.db", "wb") as f:
                        f.write(db_file.read())
                conn = sqlite3.connect("tickets.db")
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = 'gold'")
                result = cursor.fetchone()[0]
                conn.close()
                os.remove("tickets.db")
                return str(result) if result else "0"
    return "Database not found."

# Function Mapping
TASK_FUNCTIONS = {
    "readme hash": hash_prettier_readme,
    "google sheets formula": compute_google_sheets_formula,
    "wednesdays": count_weekdays,
    "csv answer": extract_csv_answer,
    "sort json": sort_json_objects,
    "replace iitm": replace_text_and_hash,
    "gold sales": calculate_gold_ticket_sales,
}

# API Endpoint
@app.post("/api/")
async def answer_question(question: str = Form(...), file: UploadFile = None):
    file_content = await file.read() if file else None
    
    for key, func in TASK_FUNCTIONS.items():
        if key in question.lower():
            return {"answer": func(file_content) if file_content else func()}
    
    # If no function is found, use OpenAI GPT-4o-mini via Proxy
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "Answer the question directly, no explanations."}, {"role": "user", "content": question}]}
    response = requests.post(AIPROXY_URL, headers=headers, json=payload).json()
    
    if "choices" not in response:
        return {"error": f"Unexpected API response format: {response}"}

    return {"answer": response["choices"][0]["message"]["content"].strip()}
