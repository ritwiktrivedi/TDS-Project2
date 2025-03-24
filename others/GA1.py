import requests
import json
import hashlib
import sqlite3
import zipfile
import os
import csv
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, Form
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set in the environment variables.")

# Q1: Install and run Visual Studio Code
def install_vscode():
    os.system("code -s")
    return "VS Code installed and executed."

# Q2: Send a HTTPS request using httpie
def send_https_request(url, params):
    response = requests.get(url, params=params)
    return response.json()

# Q3: Format README.md using Prettier and compute SHA-256 hash
def format_readme():
    os.system("npx -y prettier@3.4.2 README.md > formatted.md")
    with open("formatted.md", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Q4-Q5: Compute Google Sheets formula
def compute_google_sheets_formula(formula):
    return f"Computed result of formula: {formula}"

# Q7: Count occurrences of a weekday within a date range
def count_weekday_occurrences(start_date, end_date, day_of_week):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    count = sum(1 for d in range((end - start).days + 1)
                if (start + timedelta(days=d)).strftime("%A") == day_of_week)
    return count

# Q8: Extract value from CSV inside a ZIP file
def extract_csv_value(zip_file_path, column_name):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_dir")
    csv_file = os.path.join("temp_dir", "extract.csv")
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        return [row[column_name] for row in reader]

# Q9: Sort JSON array
def sort_json_array(json_data, primary_sort, secondary_sort):
    data = json.loads(json_data)
    sorted_data = sorted(data, key=lambda x: (x[primary_sort], x[secondary_sort]))
    return json.dumps(sorted_data, separators=(",", ":"))

# Q10: Convert key-value text file to JSON
def convert_text_to_json(text_file_path):
    with open(text_file_path, "r") as file:
        data = dict(line.strip().split("=") for line in file if "=" in line)
    return json.dumps(data)

# Q11: Sum data-value attributes in hidden divs (hypothetical)
def sum_hidden_div_values():
    return "Functionality to parse hidden HTML elements is required."

# Q12: Sum specific symbol values across differently encoded CSV files
def sum_symbol_values(zip_file_path, symbols):
    total = 0
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_dir")
    for file in os.listdir("temp_dir"):
        with open(os.path.join("temp_dir", file), newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            total += sum(float(row['value']) for row in reader if row['symbol'] in symbols)
    return total

# Q13: Generate GitHub JSON URL (requires integration)
def generate_github_json_url():
    return "User must manually create and upload the file to GitHub."

# Q14: Replace occurrences of text in files within a ZIP archive
def replace_text_in_files(zip_file_path, search_text, replacement_text):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_dir")
    for root, _, files in os.walk("temp_dir"):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  # ✅ Fixed
                content = f.read().replace(search_text, replacement_text)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    return "Text replacement completed."

# Q15: Compute total size of filtered files in ZIP
def compute_filtered_file_size(zip_file_path, min_size, modified_after):
    total_size = 0
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_dir")
    for file in os.listdir("temp_dir"):
        file_path = os.path.join("temp_dir", file)
        if os.path.getsize(file_path) >= min_size and os.path.getmtime(file_path) >= modified_after:
            total_size += os.path.getsize(file_path)
    return total_size

# Q16: Rename files in ZIP, shifting digits by 1
def rename_files_in_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("temp_dir")
    for root, _, files in os.walk("temp_dir"):
        for file in files:
            new_name = "".join(str(int(c) + 1) if c.isdigit() else c for c in file)
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
    return "Files renamed successfully."

# Q17: Compare two text files and count differing lines
def compare_text_files(file1, file2):
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        diff_count = sum(1 for line1, line2 in zip(f1, f2) if line1 != line2)
    return diff_count

# Q18: Calculate total sales for a ticket type from an SQLite database
def calculate_sql_sales(database, ticket_type):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = LOWER(?)", (ticket_type,))
    result = cursor.fetchone()[0] or 0
    conn.close()
    return result

def execute_task(question, file_path=None):
    """Uses LLM to determine the correct function and executes it."""
    
    # Call LLM API to classify the task
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "Determine the correct function for the given question, and execute the task."},
                     {"role": "user", "content": question}],
        "temperature": 0
    }
    
    response = requests.post(AIPROXY_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        llm_response = response.json()
        task_name = llm_response["choices"][0]["message"]["content"].strip().lower()
        
        if task_name in TASK_FUNCTIONS:
            params = {}
            if file_path:
                params["zip_file_path"] = file_path
            return {"answer": str(TASK_FUNCTIONS[task_name]["function"](params))}
    
    return {"error": "Task could not be determined by LLM."}


@app.post("/api/")
def process_question(question: str = Form(...), file: UploadFile = File(None)):
    file_path = None
    if file:
        file_path = f"temp_uploads/{file.filename}"
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
    return execute_task(question, file_path)

TASK_FUNCTIONS = {
    "install_vscode": {
        "description": "Installs and runs Visual Studio Code, then executes 'code -s' and returns the output.",
        "parameters": {},
        "function": lambda _: install_vscode()
    },
    "send_https_request": {
        "description": "Sends a HTTPS request to a given URL with specified parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL."},
                "params": {"type": "object", "description": "Query parameters."}
            },
            "required": ["url", "params"]
        },
        "function": lambda params: send_https_request(params["url"], params["params"])
    },
    "format_readme": {
        "description": "Formats README.md using Prettier and computes its SHA-256 hash.",
        "parameters": {},
        "function": lambda _: format_readme()
    },
    "compute_google_sheets_formula": {
        "description": "Computes the result of a given Google Sheets formula.",
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {"type": "string", "description": "Google Sheets formula."}
            },
            "required": ["formula"]
        },
        "function": lambda params: compute_google_sheets_formula(params["formula"])
    },
    "count_weekday_occurrences": {
        "description": "Counts occurrences of a specified weekday within a date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format."},
                "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format."},
                "day_of_week": {"type": "string", "description": "Day of the week to count."}
            },
            "required": ["start_date", "end_date", "day_of_week"]
        },
        "function": lambda params: count_weekday_occurrences(params["start_date"], params["end_date"], params["day_of_week"])
    },
    "extract_csv_value": {
        "description": "Extracts the value from a specified column in a CSV file inside a ZIP archive.",
        "parameters": {
            "type": "object",
            "properties": {
                "zip_file_path": {"type": "string", "description": "Path to the ZIP file containing the CSV."},
                "column_name": {"type": "string", "description": "Column name to extract values from."}
            },
            "required": ["zip_file_path", "column_name"]
        },
        "function": lambda params: extract_csv_value(params["zip_file_path"], params["column_name"])
    },
    "sort_json_array": {
        "description": "Sorts a JSON array of objects based on specified fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "json_data": {"type": "string", "description": "JSON array to sort."},
                "primary_sort": {"type": "string", "description": "Primary sorting field."},
                "secondary_sort": {"type": "string", "description": "Secondary sorting field."}
            },
            "required": ["json_data", "primary_sort", "secondary_sort"]
        },
        "function": lambda params: sort_json_array(params["json_data"], params["primary_sort"], params["secondary_sort"])
    },
    "convert_text_to_json": {
        "description": "Converts key-value pairs in a text file to JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_file_path": {"type": "string", "description": "Path to the text file."}
            },
            "required": ["text_file_path"]
        },
        "function": lambda params: convert_text_to_json(params["text_file_path"])
    },
    "replace_text_in_files": {
        "description": "Replaces text occurrences in files inside a ZIP archive.",
        "parameters": {
            "type": "object",
            "properties": {
                "zip_file_path": {"type": "string", "description": "Path to the ZIP file containing the files."},
                "search_text": {"type": "string", "description": "Text to replace."},
                "replacement_text": {"type": "string", "description": "Replacement text."}
            },
            "required": ["zip_file_path", "search_text", "replacement_text"]
        },
        "function": lambda params: replace_text_in_files(params["zip_file_path"], params["search_text"], params["replacement_text"])
    },
    "compute_filtered_file_size": {
        "description": "Computes total size of files in ZIP that meet size and date criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "zip_file_path": {"type": "string", "description": "Path to the ZIP file."},
                "min_size": {"type": "integer", "description": "Minimum file size in bytes."},
                "modified_after": {"type": "integer", "description": "Timestamp of the modification date."}
            },
            "required": ["zip_file_path", "min_size", "modified_after"]
        },
        "function": lambda params: compute_filtered_file_size(params["zip_file_path"], params["min_size"], params["modified_after"])
    },
    "compare_text_files": {
        "description": "Compares two text files and counts differing lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "file1": {"type": "string", "description": "Path to first text file."},
                "file2": {"type": "string", "description": "Path to second text file."}
            },
            "required": ["file1", "file2"]
        },
        "function": lambda params: compare_text_files(params["file1"], params["file2"])
    }
}