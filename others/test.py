from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile
import subprocess
import json
import requests
import os
import pandas as pd
import os
import re
import subprocess
import shutil
from datetime import datetime, timedelta
import re
import tempfile


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function GA1Q1: Handle Visual Studio Code command output
def solve_vscode_question(question, file=None):
    """Executes 'code -s' in the terminal and returns the output."""
    try:
        result = subprocess.run(["code", "-s"], capture_output=True, text=True)
        return {"answer": result.stdout.strip()}
    except Exception as e:
        return {"error": f"Failed to execute command: {str(e)}"}

# Function GA1Q2: Handle HTTP request using httpie
import requests

def solve_http_request_question(question, file=None):
    try:
        # Define the target URL
        url = "https://httpbin.org/get"

        # Define the parameters (URL encoded automatically by requests)
        params = {"email": "24ds2000125@ds.study.iitm.ac.in"}

        # Set custom headers to mimic HTTPie behavior
        headers = {"User-Agent": "HTTPie/3.2.4", "Accept": "*/*"}

        # Send the GET request
        response = requests.get(url, params=params, headers=headers)

        # Check if the request was successful
        response.raise_for_status()

        # Extract only the JSON body
        json_response = response.json()

        # Return formatted response inside "answer"
        return {"answer": json_response}

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to send HTTP request: {str(e)}"}


# Function GA1Q3: Handle Prettier formatting and SHA256 hashing
import os
import re
import subprocess
import shutil
import tempfile

def solve_prettier_hash_question(question, file=None):
    # Extract the file name dynamically from the question
    match = re.search(r"Download (.*?) In the directory", question)
    if not match:
        return {"error": "Could not extract the file name from the question. Ensure the question format is correct."}

    expected_filename = match.group(1).strip()  # Extracted file name from question

    if not file:
        return {"error": f"No file uploaded. Expected: {expected_filename}"}

    try:
        # Use a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, file.filename)

            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ensure the uploaded file matches the expected name
            if file.filename.lower() != expected_filename.lower():
                return {"error": f"Uploaded file '{file.filename}' does not match the expected '{expected_filename}'"}

            # Run Prettier formatting (modify the file in place)
            prettier_result = subprocess.run(
                ["npx", "-y", "prettier@3.4.2", "--write", file_path],
                capture_output=True, text=True
            )

            # Check if Prettier failed
            if prettier_result.returncode != 0:
                return {"error": f"Prettier formatting failed: {prettier_result.stderr}"}

            # Ensure the formatted file exists
            if not os.path.exists(file_path):
                return {"error": "Formatted file not found. Prettier may have failed."}

            # Compute SHA256 hash using `sha256sum`
            sha256_result = subprocess.run(
                f"sha256sum {file_path}",
                capture_output=True, text=True, shell=True
            )

            # Extract the hash value from the command output
            sha256_output = sha256_result.stdout.strip().split(" ")[0] if sha256_result.stdout else "Error computing hash"

            return {"answer": sha256_output}

    except Exception as e:
        return {"error": f"Failed to process file '{expected_filename}': {str(e)}"}


# Function GA1Q4: Solve Google Sheets formula
import numpy as np

def sum_sequence_google(rows=100, cols=100, start=9, step=15, constrain_rows=1, constrain_cols=10):
    """
    Generates a Google Sheets-style SEQUENCE matrix with column-major order.
    """
    # Generate sequence using column-major order
    sequence_matrix = np.array([[start + (r * step) + (c * rows * step) for c in range(cols)] for r in range(rows)])

    # Extract the constrained submatrix
    constrained_matrix = sequence_matrix[:constrain_rows, :constrain_cols]

    # Compute the sum
    return int(np.sum(constrained_matrix))

def solve_google_sheets_question(question, file=None):
    """
    Extracts parameters from a Google Sheets formula and computes the result.
    """
    pattern = r"SEQUENCE\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
    constrain_pattern = r"ARRAY_CONSTRAIN\(.*?,\s*(\d+),\s*(\d+)\)"

    match = re.search(pattern, question)
    constrain_match = re.search(constrain_pattern, question)

    if not match or not constrain_match:
        return {"error": "Invalid input format. Ensure the formula contains SEQUENCE and ARRAY_CONSTRAIN."}

    try:
        rows, cols, start, step = map(int, match.groups())
        constrain_rows, constrain_cols = map(int, constrain_match.groups())

        result = sum_sequence_google(rows, cols, start, step, constrain_rows, constrain_cols)

        return {"answer": result}

    except Exception as e:
        return {"error": f"Failed to process the formula: {str(e)}"}


# Function GA1Q5: Solve Office 365 Excel formula
def solve_excel_question(question, file=None):
    # Regex pattern to extract numbers from curly braces {}
    pattern = r"\{([\d,\s]+)\}"

    # Find all number groups inside {}
    matches = re.findall(pattern, question)

    if len(matches) < 2:
        return {"error": "Invalid input format. Ensure the formula contains an array and sort order."}

    try:
        # Extract array and sort order as lists of integers
        array = list(map(int, matches[0].split(",")))
        sort_order = list(map(int, matches[1].split(",")))

        # Extract `n` (number of elements to take) using regex
        n_match = re.search(r"TAKE\(.*?,\s*(\d+)\)", question)
        n = int(n_match.group(1)) if n_match else 6  # Default to 6 if not found

        # Sort the array based on the sort order
        sorted_array = [x for _, x in sorted(zip(sort_order, array))]

        # Extract the first `n` elements
        extracted_values = sorted_array[:n]

        # Compute the sum of extracted values
        return {"answer": sum(extracted_values)}

    except Exception as e:
        return {"error": f"Failed to process the formula: {str(e)}"}

# Function GA1Q6: Solve HTML hidden input question
from bs4 import BeautifulSoup

def solve_hidden_input_question(question, file=None):
    """
    Extracts the value of a hidden input field from an HTML file.
    """
    if not file:
        return {"error": "No HTML file uploaded. Please upload an HTML file containing a hidden input field."}

    try:
        # Read and parse the HTML file
        html_content = file.file.read().decode("utf-8")
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the first hidden input field
        hidden_input = soup.find("input", {"type": "hidden"})

        if hidden_input and hidden_input.get("value"):
            return {"hidden_value": hidden_input["value"]}

        return {"error": "No hidden input field with a value found in the uploaded file."}

    except Exception as e:
        return {"error": f"Failed to extract hidden input: {str(e)}"}

# Function GA1Q7: Solve count Wednesdays question
def solve_count_wednesdays_question(question, file=None):
    # Extract date range using regex
    match = re.search(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", question)
    if not match:
        return {"error": "Invalid date range format. Ensure the question contains dates in YYYY-MM-DD format."}

    try:
        # Parse the extracted start and end dates
        start_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        end_date = datetime.strptime(match.group(2), "%Y-%m-%d")

        # Initialize count
        count = 0

        # Loop through the date range
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 2:  # 2 corresponds to Wednesday
                count += 1
            current_date += timedelta(days=1)

        return {"answer": count}

    except Exception as e:
        return {"error": f"Failed to compute Wednesdays count: {str(e)}"}

# Function GA1Q8: Solve CSV extraction question
import zipfile
import pandas as pd
import os
import tempfile

def solve_csv_extraction_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing extract.csv."}

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded file temporarily
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Find and read extract.csv
            csv_path = os.path.join(tmpdirname, "extract.csv")
            if not os.path.exists(csv_path):
                return {"error": "No extract.csv found in the ZIP file."}

            df = pd.read_csv(csv_path)

            # Ensure "answer" column exists
            if "answer" not in df.columns:
                return {"error": "The 'answer' column is missing in the CSV file."}

            # Get the first non-null value in the answer column
            answer_value = df["answer"].dropna().iloc[0]

            return {"answer": str(answer_value)}

    except Exception as e:
        return {"error": f"Failed to process CSV file: {str(e)}"}

# Function GA1Q9: Solve JSON sorting question
import json
import re

def solve_json_sorting_question(question, file=None):
    """
    Sorts a JSON array by age (ascending). If ages are equal, sorts by name (alphabetically).
    Returns the sorted list inside the "answer" field.
    """
    try:
        # Extract JSON from the question
        match = re.search(r"\[.*\]", question, re.DOTALL)
        if not match:
            return {"error": "No valid JSON found in the question."}

        json_data = json.loads(match.group(0))

        # Sort by age, then by name
        sorted_data = sorted(json_data, key=lambda x: (x["age"], x["name"]))

        # Return the sorted list inside the "answer" field
        return {"answer": sorted_data}

    except Exception as e:
        return {"error": f"Failed to sort JSON data: {str(e)}"}


# Function GA1Q10: Solve JSON conversion question
import json
import hashlib

def solve_json_conversion_question(question, file=None):
    """
    Converts a text file with key=value pairs into a JSON object and computes its SHA256 hash.
    """
    if not file:
        return {"error": "No text file uploaded. Please upload a .txt file containing key=value pairs."}

    # Ensure the uploaded file is a .txt file
    if not file.filename.endswith(".txt"):
        return {"error": f"Invalid file type: {file.filename}. Please upload a .txt file."}

    try:
        # Read and process the text file
        text_content = file.file.read().decode("utf-8").strip()

        # Convert lines into key-value pairs
        key_value_pairs = {}
        for line in text_content.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)  # Only split on the first '='
                key_value_pairs[key.strip()] = value.strip()

        # Convert to compact JSON format
        json_output = json.dumps(key_value_pairs, separators=(',', ':'), sort_keys=True)

        # Compute SHA256 hash
        json_hash = hashlib.sha256(json_output.encode()).hexdigest()

        return {
            "answer": {
                "json_output": key_value_pairs,  # Properly formatted JSON object
                "sha256_hash": json_hash
            }
        }

    except Exception as e:
        return {"error": f"Failed to convert text to JSON: {str(e)}"}

# Function GA1Q11: Solve div sum question
from bs4 import BeautifulSoup

def solve_div_sum_question(question, file=None):
    if not file:
        return {"error": "No HTML file uploaded. Please upload an HTML file containing div elements."}

    try:
        # Read and parse the HTML file
        html_content = file.file.read().decode("utf-8")

        # Debugging: Print file content
        print("Received HTML Content:\n", html_content)

        soup = BeautifulSoup(html_content, "html.parser")

        # Find all <div> elements with class 'foo'
        divs_with_foo = soup.find_all("div", class_="foo")

        if not divs_with_foo:
            return {"error": "No <div> elements with class 'foo' found in the uploaded HTML file."}

        # Sum up their 'data-value' attributes
        total_sum = sum(
            int(div.get("data-value", 0)) for div in divs_with_foo if div.get("data-value", "").isdigit()
        )

        return {"answer": total_sum}

    except Exception as e:
        return {"error": f"Failed to process HTML file: {str(e)}"}

# Function GA1Q12: Solve file encoding sum question
import zipfile
import os
import tempfile
import pandas as pd

def solve_file_encoding_sum_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing data1.csv, data2.csv, and data3.txt."}

    try:
        # Create a temporary directory to extract ZIP contents
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded file temporarily
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Define the expected files and their encodings
            expected_files = {
                "data1.csv": "cp1252",
                "data2.csv": "utf-8",
                "data3.txt": "utf-16"
            }

            # Target symbols to sum values for
            target_symbols = {"š", "œ", "Ÿ"}
            total_sum = 0

            for filename, encoding in expected_files.items():
                file_path = os.path.join(tmpdirname, filename)

                # Ensure the required file exists
                if not os.path.exists(file_path):
                    return {"error": f"Missing required file: {filename} in ZIP."}

                # Read the file based on its encoding and delimiter
                if filename.endswith(".csv"):
                    df = pd.read_csv(file_path, encoding=encoding)
                else:  # Tab-separated TXT file
                    df = pd.read_csv(file_path, encoding=encoding, sep="\t")

                # Ensure required columns exist
                if "symbol" not in df.columns or "value" not in df.columns:
                    return {"error": f"File {filename} does not contain 'symbol' and 'value' columns."}

                # Convert 'value' column to numeric, ignoring errors
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

                # Sum up the values for matching symbols
                total_sum += df[df["symbol"].isin(target_symbols)]["value"].sum()

            return {"answer": total_sum}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q13: Solve GitHub repository question
import requests
import json

# Set your GitHub credentials
GITHUB_USERNAME = "your_github_username"
GITHUB_TOKEN = "your_github_token"  # Must be replaced with an actual GitHub Personal Access Token
REPO_NAME = "email-json-repo"
FILE_NAME = "email.json"
FILE_CONTENT = {"email": "24ds2000125@ds.study.iitm.ac.in"}
BRANCH_NAME = "main"

def solve_github_repo_question(question, file=None):
    try:
        # GitHub API Headers
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

        # Step 1: Create GitHub repository (if not exists)
        repo_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}"
        response = requests.get(repo_url, headers=headers)

        if response.status_code == 404:  # Repo does not exist
            create_repo_url = "https://api.github.com/user/repos"
            repo_data = {"name": REPO_NAME, "private": False}
            create_response = requests.post(create_repo_url, headers=headers, json=repo_data)
            if create_response.status_code not in [200, 201]:
                return {"error": f"Failed to create repository: {create_response.json()}"}

        # Step 2: Upload email.json file
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH_NAME}/{FILE_NAME}"
        file_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{FILE_NAME}"

        # Convert JSON content to base64
        encoded_content = json.dumps(FILE_CONTENT, indent=4)

        # Commit the file
        commit_data = {
            "message": "Added email.json",
            "content": encoded_content.encode("utf-8").decode("utf-8"),
            "branch": BRANCH_NAME
        }
        commit_response = requests.put(file_url, headers=headers, json=commit_data)

        if commit_response.status_code not in [200, 201]:
            return {"error": f"Failed to commit file: {commit_response.json()}"}

        return {"answer": raw_url}

    except Exception as e:
        return {"error": f"GitHub API request failed: {str(e)}"}

# Function GA1Q14: Solve text replacement question
import zipfile
import os
import tempfile
import hashlib
import re

def solve_replace_text_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing text files."}

    try:
        # Create a temporary directory to work with extracted files
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Process each file inside the extracted folder
            for root, _, files in os.walk(tmpdirname):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Replace all variations of 'IITM' with 'IIT Madras'
                    modified_content = re.sub(r"\bIITM\b", "IIT Madras", content, flags=re.IGNORECASE)

                    # Write back the modified content
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(modified_content)

            # Compute SHA256 checksum of concatenated file contents
            sha256_hash = hashlib.sha256()
            for root, _, files in os.walk(tmpdirname):
                for filename in sorted(files):  # Ensure consistent ordering
                    file_path = os.path.join(root, filename)
                    with open(file_path, "rb") as f:
                        while chunk := f.read(8192):
                            sha256_hash.update(chunk)

            # Return the computed SHA256 hash
            return {"answer": sha256_hash.hexdigest()}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q15: Solve file size filter question
import zipfile
import os
import tempfile
import re
from datetime import datetime, timezone, timedelta

def solve_file_size_filter_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing files."}

    try:
        # Extract minimum size requirement from the question
        size_match = re.search(r"at least (\d+) bytes", question)
        min_size = int(size_match.group(1)) if size_match else 0  # Default to 0 if not found

        # Extract the date-time requirement from the question
        date_match = re.search(r"on or after (.+)", question)
        if not date_match:
            return {"error": "Could not extract the required date-time from the question."}
        
        date_str = date_match.group(1).strip()
        
        try:
            # Convert extracted date to a comparable datetime object in IST
            IST_OFFSET = timedelta(hours=5, minutes=30)
            threshold_date = datetime.strptime(date_str, "%a, %d %b, %Y, %I:%M %p IST") - IST_OFFSET
            threshold_date = threshold_date.replace(tzinfo=timezone.utc)
        except ValueError:
            return {"error": "Invalid date format. Expected format: 'Sun, 24 Jul, 2011, 10:43 pm IST'."}

        # Create a temporary directory to extract ZIP contents
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            total_size = 0  # Track total size of matching files

            # Iterate over extracted files and apply conditions
            for root, _, files in os.walk(tmpdirname):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Get file size
                    file_size = os.path.getsize(file_path)

                    # Get modification time (convert to UTC for comparison)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path), timezone.utc)

                    # Apply conditions: size ≥ extracted min_size and modified on/after extracted threshold_date
                    if file_size >= min_size and mod_time >= threshold_date:
                        total_size += file_size

            return {"answer": total_size}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}


question_keywords = {
    "vscode": (["visual studio code", "code -s", "install vscode", "terminal command"], solve_vscode_question),
    "http_request": (["uv run", "httpie", "https request", "httpbin.org"], solve_http_request_question),
    "prettier_hash": (["npx", "prettier", "sha256sum", "README.md", "format code"], solve_prettier_hash_question),
    "google_sheets": (["google sheets", "SUM", "ARRAY_CONSTRAIN", "SEQUENCE", "spreadsheet"], solve_google_sheets_question),
    "excel": (["office 365", "SUM", "SORTBY", "TAKE", "excel formula"], solve_excel_question),
    "hidden_input": (["hidden input", "secret value", "HTML hidden field"], solve_hidden_input_question),
    "count_wednesdays": (["wednesdays", "date range", "count weekdays", "calendar calculation"], solve_count_wednesdays_question),
    "csv_extraction": (["download", "unzip", "csv file", "answer column", "extract data"], solve_csv_extraction_question),
    "json_sorting": (["sort", "JSON array", "age field", "name field", "sorting JSON"], solve_json_sorting_question),
    "json_conversion": (["multi-cursors", "key=value", "jsonhash", "convert to JSON"], solve_json_conversion_question),
    "div_sum": (["<div>", "foo class", "data-value", "sum", "hidden element"], solve_div_sum_question),
    "file_encoding_sum": (["encoding analysis", "CP-1252", "UTF-8", "UTF-16", "symbol sum"], solve_file_encoding_sum_question),
    "github_repo": (["github", "repository", "email.json", "raw url", "commit", "push"], solve_github_repo_question),
    "replace_text": (["unzip", "replace", "IITM", "IIT Madras", "sha256sum"], solve_replace_text_question),
    "file_size_filter": (["extract", "ls", "file size", "modified on", "bytes"], solve_file_size_filter_question),
}


@app.post("/api/")
async def solve_question(question: str = Form(...), file: UploadFile = File(None)):
    """
    Accepts a question and an optional file for processing.
    Dynamically selects the appropriate function based on keywords.
    """
    question_lower = question.lower()

    # Identify the category by checking for keywords
    for category, (keywords, func) in question_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return func(question, file)  # Directly call the function
    
    return {"error": "No relevant solution found. Please refine your question."}

@app.get("/")
def read_root():
    return {"message": "Hello World"}
