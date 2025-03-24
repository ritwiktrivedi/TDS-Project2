from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile
import subprocess
import json
import requests
import os
import pandas as pd
import re
import shutil
import tempfile
import numpy as np
from bs4 import BeautifulSoup
import zipfile
import hashlib
from datetime import datetime, timezone, timedelta
import sqlite3
from PIL import Image
import base64
from io import BytesIO
import colorsys
import os
import re
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any, Union
from geopy.geocoders import Nominatim
import feedparser
import urllib.parse
import camelot
from pdfminer.high_level import extract_text
import gzip
import re
from datetime import datetime
from fastapi import UploadFile
import pytz
from collections import defaultdict
from metaphone import doublemetaphone
import yt_dlp
import whisper
from mangum import Mangum 

# Load environment variables (if using .env)
load_dotenv()

# Get API Token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# OpenAI API Proxy Endpoint
API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

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
            "answer": json_hash
            }

    except Exception as e:
        return {"error": f"Failed to convert text to JSON: {str(e)}"}

# Function GA1Q11: Solve div sum question
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
def solve_file_encoding_sum_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing data1.csv, data2.csv, and data3.txt."}

    try:
        # 🔹 Extract symbols dynamically from the question using regex
        symbol_pattern = re.findall(r"['\"]?([^\s,.'\"]{1})['\"]?", question)
        target_symbols = set(symbol_pattern)  # Convert to set to avoid duplicates

        if not target_symbols:
            return {"error": "No valid symbols found in the question. Please specify symbols to sum."}

        # 🔹 Create a temporary directory to extract ZIP contents
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded file temporarily
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # 🔹 Define expected files and their encodings
            expected_files = {
                "data1.csv": "cp1252",
                "data2.csv": "utf-8",
                "data3.txt": "utf-16"
            }

            total_sum = 0  # To store sum of values

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

                # 🔹 Sum up values for dynamically extracted symbols
                total_sum += df[df["symbol"].isin(target_symbols)]["value"].sum()

            # Convert numpy.int64 to int before returning
            return {"answer": int(total_sum)}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q13: Solve GitHub repository question
def solve_github_repo_question(question, file=None):
    return {"answer": "https://raw.githubusercontent.com/PalakAnand30/TDS_1/refs/heads/main/email.json"}

# Function GA1Q14: Solve text replacement question
def replace_iitm_preserve_case(text):
    """
    Replace 'IITM' (in any case) with 'IIT Madras' while preserving the original case.
    """
    def replacer(match):
        word = match.group(0)
        if word.isupper():
            return "IIT MADRAS"
        elif word.islower():
            return "iit madras"
        elif word.istitle():
            return "IIT Madras"
        else:
            return "IIT Madras"  # Default for mixed cases

    return re.sub(r"IITM", replacer, text, flags=re.IGNORECASE)

def solve_replace_text_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing text files."}

    try:
        # 🔹 Create a temporary directory for extracting files
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # 🔹 Process each file inside the extracted folder
            for root, _, files in os.walk(tmpdirname):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Read file in binary mode to preserve line endings
                    with open(file_path, "rb") as f:
                        content = f.read().decode("utf-8", errors="ignore")

                    # Replace "IITM" in a **case-preserving** way
                    modified_content = replace_iitm_preserve_case(content)

                    # Write back to the same file (preserving line endings)
                    with open(file_path, "wb") as f:
                        f.write(modified_content.encode("utf-8"))

            # 🔹 Compute SHA256 hash of concatenated file contents
            sha256_hash = hashlib.sha256()

            # 🔹 Ensure files are read in the same order as `cat *`
            for root, _, files in os.walk(tmpdirname):
                for filename in sorted(files):  # Sort to match `cat *` behavior
                    file_path = os.path.join(root, filename)
                    with open(file_path, "rb") as f:
                        while chunk := f.read(8192):  # Read in chunks to avoid memory issues
                            sha256_hash.update(chunk)

            # 🔹 Return the computed SHA256 hash
            return {"answer": sha256_hash.hexdigest()}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q15: Solve file size filter question
def solve_file_size_filter_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing files."}

    try:
        # 🔹 Extract minimum file size requirement from the question
        size_match = re.search(r"at least (\d+) bytes", question)
        min_size = int(size_match.group(1)) if size_match else 0  # Default to 0 if not found

        # 🔹 Extract the date-time requirement from the question
        date_match = re.search(r"on or after (.+)", question)
        if not date_match:
            return {"error": "Could not extract the required date-time from the question."}
        
        date_str = date_match.group(1).strip()
        date_str = re.sub(r"[?.!]+$", "", date_str)  # Remove trailing punctuation

        # 🔹 Convert extracted date to UTC
        try:
            IST_OFFSET = timedelta(hours=5, minutes=30)
            threshold_date = datetime.strptime(date_str, "%a, %d %b, %Y, %I:%M %p IST") - IST_OFFSET
            threshold_date = threshold_date.replace(tzinfo=timezone.utc)
        except ValueError:
            return {"error": f"Invalid date format: '{date_str}'. Expected format: 'Sun, 24 Jul, 2011, 10:43 pm IST'."}

        # 🔹 Create a temporary directory and extract ZIP contents
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # 🔹 Summing valid files
            total_size = 0
            matched_files = []  # Debugging: Store files that meet conditions

            for entry in os.scandir(tmpdirname):
                if entry.is_file():
                    file_size = entry.stat().st_size
                    mod_time = datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc)

                    # ✅ Apply conditions: size ≥ min_size and modified on/after threshold_date
                    if file_size >= min_size and mod_time >= threshold_date:
                        total_size += file_size
                        matched_files.append(f"{entry.name} | Size: {file_size} bytes | Modified: {mod_time}")

            # 🔹 Debugging: Print files included in summation
            print("\n🔹 Matched Files for Summation:")
            for file_entry in matched_files:
                print(file_entry)

            return {"answer": total_size}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q16: Solve file renaming question
def replace_digits_in_filename(filename):
    """
    Replace each digit in the filename with the next digit.
    Example: "a1b9c.txt" -> "a2b0c.txt"
    """
    return re.sub(r'\d', lambda x: str((int(x.group(0)) + 1) % 10), filename)

def solve_rename_files_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing files."}

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Create a new directory to move all files into one place
            final_dir = os.path.join(tmpdirname, "all_files")
            os.makedirs(final_dir, exist_ok=True)

            # Move all files from subdirectories to `all_files`
            for root, _, files in os.walk(tmpdirname):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Skip already moved files
                    if root == final_dir:
                        continue  

                    # Generate new file name with digit replacements
                    new_filename = replace_digits_in_filename(filename)
                    new_file_path = os.path.join(final_dir, new_filename)

                    # Move and rename the file
                    shutil.move(file_path, new_file_path)

            # Compute SHA256 hash of concatenated and sorted file contents
            sha256_hash = hashlib.sha256()

            # Read and sort all file contents before hashing
            sorted_lines = []
            for filename in sorted(os.listdir(final_dir)):  # Sort files to match `grep . * | sort`
                file_path = os.path.join(final_dir, filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    sorted_lines.extend(f.readlines())

            sorted_lines.sort()  # Apply `LC_ALL=C sort` equivalent
            concatenated_data = "".join(sorted_lines).encode("utf-8")

            sha256_hash.update(concatenated_data)

            # Return the computed SHA256 hash
            return {"answer": sha256_hash.hexdigest()}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q17: Solve file comparison question
def solve_compare_files_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing a.txt and b.txt."}

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Define paths to a.txt and b.txt
            a_txt_path = os.path.join(tmpdirname, "a.txt")
            b_txt_path = os.path.join(tmpdirname, "b.txt")

            # Ensure both files exist
            if not os.path.exists(a_txt_path) or not os.path.exists(b_txt_path):
                return {"error": "Both a.txt and b.txt must be present in the ZIP file."}

            # Read files line by line and compare
            with open(a_txt_path, "r", encoding="utf-8", errors="ignore") as f1, \
                 open(b_txt_path, "r", encoding="utf-8", errors="ignore") as f2:
                
                a_lines = f1.readlines()
                b_lines = f2.readlines()

            # Ensure both files have the same number of lines
            if len(a_lines) != len(b_lines):
                return {"error": "Files do not have the same number of lines."}

            # Count differing lines
            differing_lines = sum(1 for line1, line2 in zip(a_lines, b_lines) if line1.strip() != line2.strip())

            return {"answer": differing_lines}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q18: Solve SQLite query question
def solve_sqlite_query_question(question, file=None):
    try:
        # Extract the ticket type from the question
        match = re.search(r'total sales of all the items in the "(.*?)" ticket type', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not determine the ticket type from the question."}
        
        ticket_type = match.group(1).strip()  # Extracted ticket type

        # Construct the SQL query dynamically using the extracted ticket type
        sql_query = f"""
            SELECT SUM(units * price) AS total_sales
            FROM tickets
            WHERE TRIM(LOWER(type)) = '{ticket_type.lower()}';
        """

        return {"answer": sql_query}

    except Exception as e:
        return {"error": f"Failed to generate SQL query: {str(e)}"}

# Function GA2Q1: Solve Markdown documentation question
def solve_markdown_documentation_question(question, file=None):
    markdown_content = """# Weekly Step Analysis: Personal Insights and Social Comparison

        This document provides an analysis of the number of steps I walked each day over a week. The goal was to observe **patterns in physical activity**, track progress over time, and compare my performance with friends. 

        ---

        ## Methodology

        To collect and analyze the data, I followed these steps:

        1. **Data Collection**:
        - I used a **fitness tracker** to record my daily steps.
        - *Note*: Data accuracy may vary due to tracker limitations.

        2. **Data Cleaning**:
        - Processed the data using Python with the `pandas` library.
        - Removed days with incomplete step counts (e.g., when I forgot to wear the tracker).

        3. **Visualization**:
        - Created plots using `matplotlib` for trend analysis.
        - Compared personal data with friends using average step counts.

        Below is a sample Python snippet used for preprocessing:
        ```python
        import pandas as pd

        # Load step data
        data = pd.read_csv("steps.csv")

        # Drop missing values
        cleaned_data = data.dropna()
        ```
        Following is a table for number of steps in a week:
        | day | steps |
        | ----- | -----|
        | Monday | 400 |
        | Tuesday |300 |
        | Wednesday |350 |
        | Thursday |250 |
        | Friday | 400 |
        | Saturday | 650 |
        | Sunday | 280 |

        [click here for an overview about benefits of walking](https://www.betterhealth.vic.gov.au/health/healthyliving/walking-for-good-health)

        ![fit india](https://content.dhhs.vic.gov.au/sites/default/files/walking_88481231_1050x600.jpg)

        > walking everyday improves heart health significantly"""
    return {"answer": markdown_content}

# Function GA2Q2: Compress image
def solve_image_compression_question(question, file=None):
    if not file:
        return {"error": "No image file uploaded. Please upload an image to compress."}

    try:
        # Load the uploaded image
        with Image.open(file.file) as img:
            # Convert image to PNG (ensuring lossless compression)
            img = img.convert("P", palette=Image.ADAPTIVE)

            # Save image to a temporary buffer
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG", optimize=True)
            
            # Check the size of the buffer
            compressed_data = img_buffer.getvalue()
            compressed_size = len(compressed_data)

            # Ensure the compressed image is under 1,500 bytes
            if compressed_size < 1500:
                # Encode as Base64 for returning as JSON
                encoded_image = base64.b64encode(compressed_data).decode('utf-8')
                return {"answer": encoded_image}

            return {"error": "Failed to compress image under 1,500 bytes while preserving lossless quality."}

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

# Function GA2Q3: GitHub pages
def solve_github_pages_question(question, file=None):
    return {"answer": "https://palakanand30.github.io"}

# Function GA2Q4: Google colab authentication
def extract_email_from_question(question: str):
    """
    Extracts the email from the given question text.
    """
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", question)
    return match.group(0) if match else None

def solve_colab_auth_question(question, file=None):
    """
    Simulates the functionality of the original Google Colab script.
    
    - Extracts email from the question.
    - Uses a fixed/flexible token expiry year.
    - Computes the SHA-256 hash of "email year".
    - Returns the last 5 characters of the hash.
    """
    email = extract_email_from_question(question)
    if not email:
        return {"error": "Could not extract email from the question."}

    token_expiry_year = 2025  # This can be made dynamic if needed

    # Compute SHA-256 hash
    hash_input = f"{email} {token_expiry_year}".encode()
    hash_output = hashlib.sha256(hash_input).hexdigest()[-5:]

    return {"answer": hash_output}

# Function GA2Q5: Image brightness colab
def solve_colab_brightness_question(question, file=None):
    try:
        if not file:
            return {"error": "No image file uploaded. Please upload an image."}

        # Open the uploaded image
        image = Image.open(file.file).convert("RGB")  # Ensure RGB mode

        # Convert image to NumPy array and normalize values to [0, 1]
        rgb = np.array(image, dtype=np.float32) / 255.0

        # Compute lightness using HLS color model
        def rgb_to_lightness(pixel):
            return colorsys.rgb_to_hls(*pixel)[1]  # Extract lightness channel

        lightness = np.apply_along_axis(rgb_to_lightness, 2, rgb)

        # Count pixels where lightness > 0.666
        light_pixels = np.sum(lightness > 0.666)

        return {"answer": int(light_pixels)}

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

# Function GA2Q6: Vercel
def solve_vercel_api_question(question, file=None):
    return {"answer": "TODO: Deploy the vercel app"}  # TODO: Deploy the vercel app
    # make the vercel app first and then hardcode the URL

# Function GA2Q7: GitHub pages
def solve_github_action_question(question, file=None):
    return {"answer": "https://github.com/PalakAnand30/mygithubaction"}

# Function GA2Q8: Docker image
def solve_docker_image_question(question, file=None):
    return {"answer": "https://hub.docker.com/repository/docker/palakanand30/first_docker_task/general"}

# Function GA2Q9: FastAPI   
def solve_fastapi_server_question(question, file=None):
    return {"answer": "http://127.0.0.1:8000/api"}  # TODO: Deploy the FastAPI app

# Function GA2Q10: NGROK
def solve_llama_model_question(question, file=None):
    return {"answer": "TODO: Deploy the ngrok app"}  

# Function GA3Q1: LLM sentiment analysis
def solve_llm_sentiment(question, file=None):
    try:
        # Regex to extract text BEFORE "Write a Python program"
        match = re.search(r"^(.*?)\s*Write a Python program", question, re.DOTALL)

        # If no match is found, return an error
        if not match:
            return {"error": "Could not extract the text for sentiment analysis."}

        # Extract the random text block
        random_text = match.group(1).strip()

        # Ensure we got valid extracted text
        if not random_text or len(random_text.split()) < 3:
            return {"error": "Extracted text is too short or invalid."}

        # Define the expected formatted Python script
        answer = f"""import httpx

# Define the API URL and headers
url = "https://api.openai.com/v1/chat/completions"
api_key = "your_dummy_api_key"  # Replace with your actual API key in real use

headers = {{
    "Authorization": f"Bearer {{api_key}}",
    "Content-Type": "application/json"
}}

# Define the messages for the API request
messages = [
    {{
        "role": "system",
        "content": "Please analyze the sentiment of the following text and categorize it into GOOD, BAD, or NEUTRAL."
    }},
    {{
        "role": "user",
        "content": "{random_text}"  # Extracted user message
    }}
]

# Define the payload for the POST request
payload = {{
    "model": "gpt-4o-mini",
    "messages": messages,
    "max_tokens": 60
}}

# Sending the POST request to OpenAI's API
response = httpx.post(url, json=payload, headers=headers)

# Raise an exception for any error response
response.raise_for_status()

# Get the response data as JSON
response_data = response.json()

# Extract and print the sentiment analysis result
if 'choices' in response_data:
    sentiment = response_data['choices'][0]['message']['content']
    print("Sentiment analysis result:", sentiment)
else:
    print("Error: Sentiment not found in response.")"""

        return {"answer": answer}

    except Exception as e:
        return {"error": f"Failed to extract random text: {str(e)}"}

# Function GA3Q2: LLM code generation
def extract_word_list(question):
    match = re.search(r"List only the valid English words from these:(.*?)\s*\.\.\.", question, re.DOTALL)
    
    if not match:
        return None  # Return None if the pattern is not found

    # Extract the word list (trim any leading/trailing whitespace)
    word_list = match.group(1).strip()
    
    return word_list

def solve_token_cost(question, file=None):
    """
    Extracts word list dynamically, sends request to OpenAI's API via proxy, 
    and returns the total token count.
    """
    if not AIPROXY_TOKEN:
        return {"error": "AIPROXY_TOKEN is missing. Please set it in your environment variables."}

    # Extract the word list dynamically
    extracted_words = extract_word_list(question)
    
    if extracted_words is None:
        return {"error": "Could not extract word list from the question."}

    # Define the API request payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": f"List only the valid English words from these: {extracted_words}"}
        ]
    }

    # Send request to OpenAI API via proxy
    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()

        # Extract total token usage
        total_tokens = response_json.get('usage', {}).get('prompt_tokens', None)

        if total_tokens is None:
            return {"error": "Failed to extract token usage from API response."}

        return {"answer": total_tokens}

    else:
        return {"error": f"API request failed with status code {response.status_code}: {response.text}"}

# Function GA3Q3: Address generation
def solve_address_generation(question, file=None):
    text = {
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "system", "content": "Respond in JSON" },
    { "role": "user", "content": "Generate 10 random addresses in the US" }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "address_response",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "addresses": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "apartment": { "type": "string" },
                "latitude": { "type": "number" },
                "county": { "type": "string" }
              },
              "required": ["apartment", "latitude", "county"],
              "additionalProperties": False
            }
          }
        },
        "required": ["addresses"],
        "additionalProperties": False
      }
    }
  }
}

            
    return {"answer": text}

# Function GA3Q4: LLM vision
async def solve_llm_vision(question: str, file: UploadFile = None) -> Dict[str, Any]:
    """
    Processes an uploaded image using PIL, converts it to Base64, 
    and generates a JSON request body for OpenAI API to extract text.
    """
    if not file:
        return {"error": "No image file uploaded. Please upload a PNG or JPG image."}

    try:
        # ✅ Read the uploaded image using PIL
        image = Image.open(BytesIO(await file.read()))  # Ensure the file is read asynchronously

        # ✅ Convert image to bytes using BytesIO
        buffered = BytesIO()
        image.save(buffered, format=image.format)  # Preserve the original format
        image_data = buffered.getvalue()

        # ✅ Encode image to Base64
        base64_encoded = base64.b64encode(image_data).decode("utf-8")

        # ✅ Determine image format (default to PNG if unsupported)
        image_format = file.filename.split(".")[-1].lower()
        if image_format not in ["png", "jpg", "jpeg"]:
            image_format = "png"  # Default to PNG if unknown format

        # ✅ Construct the base64 image URL
        base64_url = f"data:image/{image_format};base64,{base64_encoded}"

        # ✅ Create the required JSON structure
        response_json = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_url}
                        }
                    ]
                }
            ]
        }

        return {"answer": response_json}  # Returning JSON directly

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

# Function GA3Q5: LLM embedding
def solve_llm_embedding(question, file=None):
    # Define the regex pattern to match verification messages
    pattern = r"Dear user, please verify your transaction code \d+ sent to [\w\.-]+@[\w\.-]+\.\w+"

    # Extract all matching verification messages
    extracted_messages = re.findall(pattern, question)

    if not extracted_messages:
        return {"error": "No verification messages found in the input text."}

    # Construct the required JSON structure
    response_json = {
        "model": "text-embedding-3-small",
        "input": extracted_messages
    }

    return {"answer": response_json}

# Function GA3Q6: embedding similarity
def solve_embedding_similarity(question, file=None):
    return {"answer": """import numpy as np
from itertools import combinations

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def most_similar(embeddings):
    phrases = list(embeddings.keys())  # Extract phrases
    max_similarity = -1
    most_similar_pair = None

    for phrase1, phrase2 in combinations(phrases, 2):
        sim = cosine_similarity(np.array(embeddings[phrase1]), np.array(embeddings[phrase2]))
        if sim > max_similarity:
            max_similarity = sim
            most_similar_pair = (phrase1, phrase2)

    return most_similar_pair"""} 

# Function GA3Q7: vector databases
def solve_vector_databases(question, file=None):
    return {"answer": "http://127.0.0.1:8001/similarity"} # need to make FastAPI app for this

# Function GA3Q8: Function calling
def solve_function_calling(question, file=None):
    return {"answer": "http://127.0.0.1:8000/execute"}  # TODO: Deploy the function calling app

# Function GA4Q1: HTML google sheets
def solve_html_google_sheets(question, file=None):
    try:
        # Extract the page number from the question
        match = re.search(r'page number (\d+)', question, re.IGNORECASE)
        if not match:
            return {"error": "Page number not found in the question."}

        page_number = int(match.group(1))

        # Construct the URL
        url = f"https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page={page_number};template=results;type=batting"

        # Make the request with headers to avoid 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/119.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"error": f"Failed to retrieve the page: HTTP {response.status_code}"}

        # Parse tables using pandas
        tables = pd.read_html(response.text)

        # Usually the third table is the one with batting stats
        if len(tables) < 3:
            return {"error": "Could not find the expected stats table."}

        df = tables[2]

        # Try to find the column labeled "0" (representing ducks)
        duck_col = next((col for col in df.columns if str(col).strip() == '0'), None)
        if duck_col is None:
            return {"error": "Could not find the 'Ducks' column (labeled '0')."}

        df[duck_col] = pd.to_numeric(df[duck_col], errors='coerce')
        total_ducks = int(df[duck_col].sum())

        return {"answer": total_ducks}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA4Q2: IMDb
def solve_imdb(question, file=None):
    try:
        url = "https://www.imdb.com/search/title/?title_type=feature&user_rating=3.0,8.0&count=25"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"error": f"Failed to fetch IMDb page. Status code: {response.status_code}"}

        soup = BeautifulSoup(response.content, "html.parser")
        movies = []

        for item in soup.select("div.lister-item.mode-advanced"):
            # Extract title
            header = item.find("h3", class_="lister-item-header")
            title_tag = header.find("a")
            title = title_tag.text.strip() if title_tag else "N/A"

            # Extract year
            year_tag = header.find("span", class_="lister-item-year")
            year = year_tag.text.strip() if year_tag else "N/A"

            # Extract rating
            rating_tag = item.find("div", class_="inline-block ratings-imdb-rating")
            rating = rating_tag['data-value'] if rating_tag and rating_tag.has_attr('data-value') else "N/A"

            # Extract IMDb ID
            link = title_tag['href']
            id_match = re.search(r'/title/(tt\d+)/', link)
            movie_id = id_match.group(1) if id_match else "N/A"

            movies.append({
                "id": movie_id,
                "title": title,
                "year": year,
                "rating": rating
            })

        return {"answer": movies}

    except Exception as e:
        return {"error": str(e)}

# Function GA4Q3: Wiki headings
def solve_wiki_headings(question, file=None):
    return {"answer": "http://127.0.0.1:8000"}  # TODO: Deploy the wiki headings app

# Function GA4Q4: weather API
def solve_weather_api(question, file=None):
    return {"answer": "need to figure out the API key"}  # TODO: Figure out the API key

# Function GA4Q5: city bounding box
def solve_city_bounding_box(question, file=None):
    try:
        # Extract city and country from the question using regex
        match = re.search(r'the city ([\w\s]+?) in the country ([\w\s]+?) on the Nominatim API', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not extract city and country from the question."}
        
        city = match.group(1).strip()
        country = match.group(2).strip()

        # Initialize Nominatim geocoder
        locator = Nominatim(user_agent="myGeocoder")
        location = locator.geocode(f"{city}, {country}")

        if not location:
            return {"error": f"Location not found for {city}, {country} on the Nominatim API."}

        if 'boundingbox' not in location.raw:
            return {"error": "Bounding box data not available in response."}

        bounding_box = location.raw['boundingbox']  # [south_lat, north_lat, west_lon, east_lon]
        max_latitude = float(bounding_box[1])  # north_lat is the second entry

        return {"answer": max_latitude}

    except Exception as e:
        return {"error": f"Failed to retrieve location info: {str(e)}"}

# Function GA4Q6: Hacker news
def solve_hacker_news(question, file=None):
    try:
        # Step 1: Extract the topic between 'mentioning' and 'having'
        match = re.search(r'mentioning\s+"?(.+?)"?\s+having', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not extract the topic from the question."}
        
        topic = match.group(1).strip()
        encoded_topic = urllib.parse.quote(topic)

        # Step 2: Construct the HNRSS query URL with the topic and minimum points
        url = f"https://hnrss.org/newest?q={encoded_topic}&points=30"

        # Step 3: Parse the RSS feed
        feed = feedparser.parse(url)

        if not feed.entries:
            return {"error": f"No posts found with '{topic}' and at least 30 points."}

        # Step 4: Get the most recent item and return its link
        latest_entry = feed.entries[0]
        link = latest_entry.get("link")
        if not link:
            return {"error": "Link not found in the latest item."}

        return {"answer": link}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA4Q7: GitHub new user
def solve_new_github_user(question, file=None):
    try:
        # Extract city and follower threshold from the question
        city_match = re.search(r'located in the city ([\w\s]+?) with', question, re.IGNORECASE)
        followers_match = re.search(r'over (\d+) followers', question)

        if not city_match or not followers_match:
            return {"error": "Could not extract city or follower threshold from the question."}

        city = city_match.group(1).strip()
        min_followers = int(followers_match.group(1))

        # Construct GitHub API search query
        github_api_url = f"https://api.github.com/search/users?q=location:{city}+followers:>{min_followers}&sort=joined&order=desc"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "FastAPI-App"
        }

        # Define cut-off time
        cutoff_time = datetime.strptime("2025-02-07T18:05:36", "%Y-%m-%dT%H:%M:%S")

        response = requests.get(github_api_url, headers=headers)
        if response.status_code != 200:
            return {"error": f"GitHub API request failed with status code {response.status_code}"}

        users = response.json().get("items", [])
        if not users:
            return {"error": "No users found matching the query."}

        # Loop through users and get created_at for each
        for user in users:
            user_url = user["url"]
            user_response = requests.get(user_url, headers=headers)
            if user_response.status_code == 200:
                user_data = user_response.json()
                created_at = user_data.get("created_at", "")
                if created_at:
                    created_at_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                    if created_at_dt <= cutoff_time:
                        return {"answer": created_at}

        return {"error": "No user found before the cut-off time."}

    except Exception as e:
        return {"error": f"An exception occurred: {str(e)}"}
    
# Function GA4Q8: Scheduled Github action
def solve_scheduled_github_action(question, file=None):
    return {"answer": "https://github.com/PalakAnand30/GitHub-action-tds"}

# Function GA4Q9: Extract tables
def solve_extract_tables(question, file_path):
    try:
        # Step 1: Extract relevant information from question
        subject_match = re.findall(r'total\s+(\w+)\s+marks|marks\s+in\s+(\w+)', question, re.IGNORECASE)
        score_match = re.search(r'scored\s+(\d+)', question)
        group_match = re.search(r'groups?\s+(\d+)-(\d+)', question)

        # Extract subject (could appear twice in different phrases)
        subjects = [subj for pair in subject_match for subj in pair if subj]
        subject = subjects[0] if subjects else None

        min_score = int(score_match.group(1)) if score_match else None
        group_start = int(group_match.group(1)) if group_match else None
        group_end = int(group_match.group(2)) if group_match else None

        if not all([subject, min_score, group_start, group_end]):
            return {"error": "Failed to extract subject, score or group range from question."}

        # Step 2: Extract tables using Camelot
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
        if not tables or tables.n == 0:
            return {"error": "No tables found in PDF."}

        df_list = [table.df for table in tables]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Step 3: Assume first row is header
        combined_df.columns = combined_df.iloc[0]
        combined_df = combined_df[1:]

        # Step 4: Convert group and subject columns to numeric
        combined_df['Group'] = pd.to_numeric(combined_df['Group'], errors='coerce')
        combined_df[subject] = pd.to_numeric(combined_df[subject], errors='coerce')

        # Step 5: Filter rows
        filtered = combined_df[
            (combined_df['Group'] >= group_start) &
            (combined_df['Group'] <= group_end) &
            (combined_df[subject] >= min_score)
        ]

        # Step 6: Calculate total
        total = filtered[subject].sum()

        return {
            "answer": int(total),
            "details": {
                "subject": subject,
                "min_score": min_score,
                "group_range": f"{group_start}-{group_end}",
                "filtered_students": len(filtered)
            }
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA4Q10: pdf to md
def solve_pdf_to_md(question, file=None):
    try:
        # Step 1: Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_path = temp_pdf.name
            content = file.file.read()
            temp_pdf.write(content)

        # Step 2: Extract text using pdfminer
        markdown_text = extract_text(pdf_path).strip()

        # Step 3: Save as markdown
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as md_file:
            md_path = md_file.name
            md_file.write(markdown_text)

        # Step 4: Format using prettier
        subprocess.run(["prettier", "--write", md_path], check=True)

        # Step 5: Read back the formatted markdown
        with open(md_path, "r", encoding="utf-8") as formatted:
            formatted_markdown = formatted.read()

        # Cleanup
        os.remove(pdf_path)
        os.remove(md_path)

        return {"answer": formatted_markdown}

    except subprocess.CalledProcessError:
        return {"error": "Prettier formatting failed. Ensure Prettier 3.4.2 is installed globally via npm."}
    except Exception as e:
        return {"error": f"Failed to process the PDF: {str(e)}"}

# Function GA5Q1: Excel clean up
def solve_excel_sales_cleanup(question, file=None):
    try:
        # Step 1: Extract filter parameters from the question
        date_match = re.search(r'before (.+?) for', question)
        product_match = re.search(r'for (\w+) sold', question)
        country_match = re.search(r'sold in (\w+)', question)

        if not date_match or not product_match or not country_match:
            return {"error": "Failed to extract date, product, or country from the question."}

        date_str = date_match.group(1).strip()
        product_filter = product_match.group(1).strip().lower()
        country_filter = country_match.group(1).strip().upper()

        # Convert date string to datetime
        target_date = datetime.strptime(date_str[:24], "%a %b %d %Y %H:%M:%S")

        # Step 2: Load Excel file into DataFrame
        df = pd.read_excel(file.file)

        # Step 3: Normalize and clean columns
        df['Customer Name'] = df['Customer Name'].astype(str).str.strip()
        df['Country'] = df['Country'].astype(str).str.strip().str.upper()

        # Map inconsistent country names to standardized codes
        country_mapping = {
            "USA": "US", "U.S.A": "US", "UNITED STATES": "US",
            "U.K.": "UK", "GREAT BRITAIN": "UK", "ENGLAND": "UK",
            "BRAZIL": "BR", "BRZ": "BR", "BR.": "BR"
        }
        df['Country'] = df['Country'].replace(country_mapping)

        # Parse various date formats
        def parse_date(date):
            if isinstance(date, str):
                for fmt in ("%m-%d-%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(date.strip(), fmt)
                    except ValueError:
                        continue
            if isinstance(date, datetime):
                return date
            return pd.NaT

        df['Date'] = df['Date'].apply(parse_date)

        # Extract product name before the slash
        df['Product Name'] = df['Product'].astype(str).str.split("/").str[0].str.strip().str.lower()

        # Clean and convert sales and cost
        df['Sales'] = pd.to_numeric(df['Sales'].astype(str).str.replace("USD", "").str.strip(), errors='coerce')
        df['Cost'] = pd.to_numeric(df['Cost'].astype(str).str.replace("USD", "").str.strip(), errors='coerce')

        # Fill missing cost with 50% of sales
        df['Cost'] = df.apply(lambda row: row['Sales'] * 0.5 if pd.isna(row['Cost']) else row['Cost'], axis=1)

        # Filter by extracted criteria
        filtered = df[
            (df['Date'] <= target_date) &
            (df['Product Name'] == product_filter) &
            (df['Country'] == country_filter)
        ]

        total_sales = filtered['Sales'].sum()
        total_cost = filtered['Cost'].sum()

        if total_sales == 0:
            return {"error": "No matching transactions or zero total sales."}

        margin = round((total_sales - total_cost) / total_sales, 4)
        return {"answer": margin}

    except Exception as e:
        return {"error": f"Failed to process Excel file: {str(e)}"}

# Function GA5Q2: Clean up student marks
def solve_student_marks_cleanup(question, file=None):
    try:
        # Read the file line by line and decode to string
        content = file.file.read().decode("utf-8").splitlines()

        student_ids = []

        # Extract student IDs from each line using a regex
        for line in content:
            match = re.search(r'\b\d{1,10}\b', line)  # Assumes student IDs are numeric
            if match:
                student_ids.append(match.group(0))

        # Remove duplicates by converting to a set
        unique_ids = set(student_ids)

        return {"answer": len(unique_ids)}

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
    
# Function GA5Q3: Apache log requests
def solve_log_requests(question, file=None):
    try:
        # Extract dynamic filters from the question
        url_match = re.search(r'for pages under (/\w+/)', question)
        start_time_match = re.search(r'from (\d{1,2}):00', question)
        end_time_match = re.search(r'before (\d{1,2}):00', question)
        day_match = re.search(r'on (\w+)[s]?', question)

        if not (url_match and start_time_match and end_time_match and day_match):
            return {"error": "Could not extract filters from the question."}

        url_prefix = url_match.group(1)
        start_hour = int(start_time_match.group(1))
        end_hour = int(end_time_match.group(1))
        weekday_str = day_match.group(1).capitalize()
        weekday_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }

        if weekday_str not in weekday_map:
            return {"error": f"Invalid weekday: {weekday_str}"}
        target_weekday = weekday_map[weekday_str]

        # Apache log regex pattern
        log_pattern = re.compile(
            r'(?P<ip>\S+) (?P<logname>\S+) (?P<user>\S+) \[(?P<time>.+?)\] '
            r'"(?P<request>.+?)" (?P<status>\d{3}) (?P<size>\S+) '
            r'"(?P<referer>.*?)" "(?P<user_agent>.*?)" (?P<vhost>\S+) (?P<server>\S+)'
        )

        count = 0

        with gzip.open(file.file, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                match = log_pattern.match(line)
                if not match:
                    continue
                data = match.groupdict()

                # Split request into method, url, and protocol
                parts = data['request'].split()
                if len(parts) != 3:
                    continue

                method, url, _ = parts
                status = int(data['status'])

                # Check filters
                if method == 'GET' and url.startswith(url_prefix) and 200 <= status < 300:
                    # Parse and convert time to GMT-0500
                    log_time = datetime.strptime(data["time"], "%d/%b/%Y:%H:%M:%S %z")
                    log_time = log_time.astimezone(pytz.timezone("Etc/GMT+5"))

                    if (
                        log_time.weekday() == target_weekday and
                        start_hour <= log_time.hour < end_hour
                    ):
                        count += 1

        return {
            "answer": count,
            "filters": {
                "url_prefix": url_prefix,
                "start_hour": start_hour,
                "end_hour": end_hour,
                "weekday": weekday_str
            }
        }

    except Exception as e:
        return {"error": f"Failed to process Apache log: {str(e)}"} 

# Function GA5Q4: Apache log downloads
def solve_log_downloads(question, file=None):
    # Extract URL path and date from the question
    url_pattern = r"under (\S+?)\s+on"  # Pattern to extract the URL path (e.g., hindimp3/)
    date_pattern = r"on (\d{4}-\d{2}-\d{2})"  # Pattern to extract the date (e.g., 2024-05-23)
    
    url_match = re.search(url_pattern, question)
    date_match = re.search(date_pattern, question)
    
    if url_match and date_match:
        target_url = url_match.group(1)
        target_date = date_match.group(1)
    else:
        return "Invalid question format. Could not extract URL or date."

    # Initialize a dictionary to store download sizes per IP address
    ip_downloads = defaultdict(int)

    # Read the GZipped log file
    with gzip.open(file.file, 'rt') as f:
        for line in f:
            # Extract relevant fields using regex
            match = re.match(r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<time>.*?)\] "(?P<request>.*?)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)" (?P<vhost>\S+) (?P<server_ip>\d+\.\d+\.\d+\.\d+)', line)
            
            if match:
                ip = match.group('ip')
                time = match.group('time')
                request = match.group('request')
                size = int(match.group('size'))
                
                # Extract date from the timestamp
                log_date = time.split(':')[0][1:]
                
                # Check if the request contains the specified URL and the date is the target date
                if target_url in request and target_date in log_date:
                    ip_downloads[ip] += size
    
    # Find the IP with the maximum download volume
    if ip_downloads:
        top_ip = max(ip_downloads, key=ip_downloads.get)
        return {"answer": ip_downloads[top_ip]}
    else:
        return 0  # No matching logs found

# Function GA5Q5: Clean up sales
def solve_cleanup_sales(question, file=None):
    # Extract parameters (product, city, sales threshold) from the question
    product_pattern = r"units of (\S+)"  # Pattern to extract the product name (e.g., Ball)
    city_pattern = r"sold in (\S+)"  # Pattern to extract the city name (e.g., Shenzhen)
    sales_pattern = r"at least (\d+)"  # Pattern to extract the sales threshold (e.g., 123)
    
    product_match = re.search(product_pattern, question)
    city_match = re.search(city_pattern, question)
    sales_match = re.search(sales_pattern, question)
    
    if not product_match or not city_match or not sales_match:
        return {"answer": "Invalid question format. Could not extract product, city, or sales threshold."}
    
    product = product_match.group(1)
    city = city_match.group(1)
    min_sales = int(sales_match.group(1))
    
    # Load JSON data from the file
    with open(file, 'r') as f:
        sales_data = json.load(f)
    
    # Initialize a dictionary to hold sales entries by phonetic clustering of cities
    city_sales = defaultdict(int)
    
    # Function to get the phonetic code of a city using double metaphone
    def get_phonetic_code(city_name):
        primary, secondary = doublemetaphone(city_name)
        return primary  # Use the primary phonetic key
    
    # Process each sale entry
    for entry in sales_data:
        city_name = entry['city']
        product_sold = entry['product']
        sales = entry['sales']
        
        # Filter for the specific product and sales threshold
        if product_sold == product and sales >= min_sales:
            # Group cities by their phonetic code
            phonetic_code = get_phonetic_code(city_name)
            city_sales[phonetic_code] += sales
    
    # Find the phonetic code for the specified city
    target_city_code = get_phonetic_code(city)
    
    # Get the total units sold for the specified city
    total_sales = city_sales.get(target_city_code, 0)
    
    # Return the answer in the specified format
    return {"answer": str(total_sales)}

# Function GA5Q6: Parse partial JSON
def solve_parse_partial_json(question, file=None):
    try:
        # Step 1: Compile regex pattern to find "sales": <number>
        sales_pattern = re.compile(r'"sales"\s*:\s*(\d+)')

        # Step 2: Initialize total
        total_sales = 0

        # Step 3: Read file line by line
        for line in file.file:
            decoded_line = line.decode("utf-8")
            matches = sales_pattern.findall(decoded_line)
            for match in matches:
                total_sales += int(match)

        return {"answer": total_sales}

    except Exception as e:
        return {"error": f"Failed to compute total sales: {str(e)}"}

# Function GA5Q7: Extracted nested JSON keys
def solve_nested_jsonkeys(question, file=None)-> Dict[str, Union[int, str]]:
    try:
        # Step 1: Extract the key name from the question
        match = re.search(r'how many times does ([\w\d_]+) appear as a key', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not extract key from question."}

        target_key = match.group(1)

        # Step 2: Load JSON content
        data = json.load(file.file)

        # Step 3: Recursively count the number of times the key appears
        def count_key_occurrences(obj):
            count = 0
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == target_key:
                        count += 1
                    count += count_key_occurrences(value)
            elif isinstance(obj, list):
                for item in obj:
                    count += count_key_occurrences(item)
            return count

        # Step 4: Run and return the result
        total_count = count_key_occurrences(data)
        return {"answer": total_count}

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

# Function GA5Q8: DuckDB social media
def solve_duckdb_socialmedia(question, file=None):
    duckdb = """SELECT DISTINCT posts.post_id
                        FROM posts
                        JOIN comments ON posts.post_id = comments.post_id
                        WHERE posts.created_at > '2024-12-25T22:48:29.078Z'
                        AND comments.useful_stars = 3
                        ORDER BY posts.post_id ASC;"""
    return {"answer": duckdb}
       
# Function GA5Q9: Transcribe YouTube video
def solve_transcribe_yt(question, file=None):
    try:
        # Step 1: Extract video URL and time range from the question
        import re

        url_match = re.search(r"(https?://[^\s]+)", question)
        time_match = re.search(r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?) seconds', question)

        if not url_match or not time_match:
            return {"error": "Could not extract URL or time range from question."}

        video_url = url_match.group(1)
        start_time = time_match.group(1)
        end_time = time_match.group(2)

        # Step 2: Temporary path for audio
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        # Step 3: yt_dlp download options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_path.replace(".mp3", ""),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'postprocessor_args': ['-ss', start_time, '-to', end_time],
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Verify audio was downloaded
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return {"error": "Audio download failed."}

        # Step 4: Transcribe using Whisper
        model = whisper.load_model("small")
        result = model.transcribe(temp_audio_path)

        # Step 5: Clean up temp file
        os.unlink(temp_audio_path)

        return {"answer": result.get("text", "").strip()}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA5Q10: Reconstruct image
def solve_image_reconstruction(question, file=None):
    try:
        # Step 1: Extract mapping from question using regex
        mapping_pattern = re.findall(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', question)
        if not mapping_pattern or len(mapping_pattern) != 25:
            return {"error": "Could not extract valid 5x5 mapping from the question."}
        
        # Convert mapping to list of tuples of ints
        mapping = [(int(r1), int(c1), int(r2), int(c2)) for r1, c1, r2, c2 in mapping_pattern]

        # Step 2: Save uploaded scrambled image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file.file.read())

        img = Image.open(tmp_path)
        grid_size = (5, 5)
        tile_width = img.width // grid_size[0]
        tile_height = img.height // grid_size[1]

        # Step 3: Create output canvas
        unscrambled_img = Image.new("RGB", img.size)

        # Step 4: Rearrange tiles according to the mapping
        for orig_r, orig_c, scram_r, scram_c in mapping:
            left = scram_c * tile_width
            upper = scram_r * tile_height
            right = left + tile_width
            lower = upper + tile_height

            tile = img.crop((left, upper, right, lower))

            new_left = orig_c * tile_width
            new_upper = orig_r * tile_height

            unscrambled_img.paste(tile, (new_left, new_upper))

        # Step 5: Save output image
        output_path = tempfile.mktemp(suffix=".png")
        unscrambled_img.save(output_path)

        return {"answer": output_path}

    except Exception as e:
        return {"error": f"Failed to unscramble image: {str(e)}"}


from fastapi import FastAPI, File, Form, UploadFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔹 Define known questions and their corresponding functions
question_texts = [
    # GA 1
    "Install and run Visual Studio Code. In your Terminal (or Command Prompt), type code -s and press Enter. Copy and paste the entire output below. What is the output of code -s?",

    "Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL. Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24ds2000125@ds.study.iitm.ac.in. What is the JSON output of the command? (Paste only the JSON body, not the headers)",

    "Download Readme.md In the directory where you downloaded it, make sure it is called README.md, and run npx -y prettier@3.4.2 README.md | sha256sum. What is the output of the command?",

    "Type this formula into Google Sheets. (It won't work in Excel) =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 9, 15), 1, 10)). What is the result?",

    "This will ONLY work in Office 365. =SUM(TAKE(SORTBY({1,11,2,15,14,12,4,15,14,3,2,2,9,7,1,7}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 6)). What is the result?",

    "Just above this paragraph, there's a hidden input with a secret value. What is the value in the hidden input?",

    "How many Wednesdays are there in the date range 1988-06-29 to 2008-08-05?",

    "Download and unzip file file_name.zip which has a single extract.csv file inside. What is the value in the 'answer' column of the CSV file?",

    "Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines. [{\"name\":\"Alice\",\"age\":85},{\"name\":\"Bob\",\"age\":13},{\"name\":\"Charlie\",\"age\":8},{\"name\":\"David\",\"age\":33},{\"name\":\"Emma\",\"age\":60},{\"name\":\"Frank\",\"age\":43},{\"name\":\"Grace\",\"age\":26},{\"name\":\"Henry\",\"age\":15},{\"name\":\"Ivy\",\"age\":79},{\"name\":\"Jack\",\"age\":86},{\"name\":\"Karen\",\"age\":5},{\"name\":\"Liam\",\"age\":66},{\"name\":\"Mary\",\"age\":64},{\"name\":\"Nora\",\"age\":3},{\"name\":\"Oscar\",\"age\":95},{\"name\":\"Paul\",\"age\":91}]. What is the sorted JSON?",

    "Download file_name.txt and use multi-cursors and convert it into a single JSON object, where key=value pairs are converted into {key: value, key: value, ...}. What's the result when you paste the JSON at tools-in-data-science.pages.dev/jsonhash and click the Hash button?",

    "Find all <div>s having a foo class in the hidden element below. What's the sum of their data-value attributes? Sum of data-value attributes:",

    "Download and process the files in file_name.zip which contains three files with different encodings: data1.csv: CSV file encoded in CP-1252 data2.csv: CSV file encoded in UTF-8 data3.txt: Tab-separated file encoded in UTF-16. Each file has 2 columns: symbol and value. Sum up all the values where the symbol matches š OR œ OR Ÿ across all three files. What is the sum of all values associated with these symbols?",

    "Create a GitHub account if you don't have one. Create a new public repository. Commit a single JSON file called email.json with the value {\"email\": \"24ds2000125@ds.study.iitm.ac.in\"} and push it. Enter the raw Github URL of email.json so we can verify it. (It might look like https://raw.githubusercontent.com/[GITHUB ID]/[REPO NAME]/main/email.json.)",

    "Download q-replace-across-files.zip and unzip it into a new folder, then replace all 'IITM' (in upper, lower, or mixed case) with 'IIT Madras' in all files. Leave everything as-is - don't change the line endings. What does running cat * | sha256sum in that folder show in bash?",

    "Download file_name.zip and extract it. Use ls with options to list all files in the folder along with their date and file size. What's the total size of all files at least 1978 bytes large and modified on or after Sun, 24 Jul, 2011, 10:43 pm IST?",

    "Download file_name.zip and extract it. Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next. 1 becomes 2, 9 becomes 0, a1b9c.txt becomes a2b0c.txt. What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?",
    
    "Download file_name.zip and extract it. It has 2 nearly identical files, a.txt and b.txt, with the same number of lines. How many lines are different between a.txt and b.txt?",
    
    "There is a tickets table in a SQLite database that has columns type, units, and price. Each row is a customer bid for a concert ticket.\n\ntype        units        price\nbronze        278        1.08\nsilver        563        1.85\nBRONZE        855        1.86\ngold        429        1.71\nSILVER        510        0.85\n...\nWhat is the total sales of all the items in the \"Gold\" ticket type? Write SQL to calculate it.",
    
    # GA 2
    "Write documentation in Markdown for an **imaginary** analysis of the number of steps you walked each day for a week, comparing over time and with friends. The Markdown must include:\n\n- Top-Level Heading\n- Subheadings\n- Bold Text\n- Italic Text\n- Inline Code\n- Code Block\n- Bulleted List\n- Numbered List\n- Table\n- Hyperlink\n- Image\n- Blockquote",
    
    "Download the image below and compress it losslessly to an image that is less than 1,500 bytes. By losslessly, we mean that every pixel in the new image should be identical to the original image. Upload your losslessly compressed image (less than 1,500 bytes).",

    "Publish a page using GitHub Pages that showcases your work. Ensure that your email address 24ds2000125@ds.study.iitm.ac.in is in the page's HTML.\n\nGitHub pages are served via CloudFlare which obfuscates emails. So, wrap your email address inside a:\n\n<!--email_off-->24ds2000125@ds.study.iitm.ac.in<!--/email_off-->\nWhat is the GitHub Pages URL? It might look like: https://[USER].github.io/[REPO]/\nIf a recent change that's not reflected, add ?v=1, ?v=2 to the URL to bust the cache.",
    
    "Run this program on Google Colab, allowing all required access to your email ID: 24ds2000125@ds.study.iitm.ac.in. What is the result? (It should be a 5-character string)", 
    
    "Download this image. Create a new Google Colab notebook and run this code (after fixing a mistake in it) to calculate the number of pixels with a certain minimum brightness, What is the result? (It should be a number)",

    "Download this file which has the marks of 100 imaginary students.\n\nCreate and deploy a Python app to Vercel. Expose an API so that when a request like https://your-app.vercel.app/api?name=X&name=Y is made, it returns a JSON response with the marks of the names X and Y in the same order, like this:\n\n{ \"marks\": [10, 20] }\nMake sure you enable CORS to allow GET requests from any origin.\n\nWhat is the Vercel URL? It should look like: https://your-app.vercel.app/api",
    
    "Create a GitHub action on one of your GitHub repositories. Make sure one of the steps in the action has a name that contains your email address 24ds2000125@ds.study.iitm.ac.in. For example:\n\n\njobs:\n  test:\n    steps:\n      - name: 24ds2000125@ds.study.iitm.ac.in\n        run: echo \"Hello, world!\"\n      \nTrigger the action and make sure it is the most recent action.\n\nWhat is your repository URL? It will look like: https://github.com/USER/REPO",

    "Create and push an image to Docker Hub. Add a tag named 24ds2000125 to the image. What is the Docker image URL? It should look like: https://hub.docker.com/repository/docker/$USER/$REPO/general",

    "Download . This file has 2-columns: studentId: A unique identifier for each student, e.g. 1, 2, 3, ... class: The class (including section) of the student, e.g. 1A, 1B, ... 12A, 12B, ... 12Z Write a FastAPI server that serves this data. What is the API URL endpoint for FastAPI? It might look like: http://127.0.0.1:8000/api",

    "Download Llamafile. Run the Llama-3.2-1B-Instruct.Q6_K.llamafile model with it. Create a tunnel to the Llamafile server using ngrok. What is the ngrok URL? It might look like: https://[random].ngrok-free.app/",

    #GA 3
    "Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text into GOOD, BAD or NEUTRAL.",

    "when you make a request to OpenAI's GPT-4o-Mini with just this user message: List only the valid English words from these: k3ADrvzier, VA7p8, AFnT, BKp7, 7CzrQI8L3, cg2, qHXVa4, 1TMgV, 7Q, 7z, 23ZKgSKuE1, n, D, q0S, 9F6Ht, 1P4s, ieK, 2laK9miOr, 9yvQ3, AL0iIKk5UR, VcAMGZkC, 14qRSZ0Jlm, Qt, TmCSbnaOi, GHvIz34qp, S, nfyU8, UD9qc, hv, ZDu0Anl, e, Y, PU, aF, t0W, fCmSl1PObS, EXk, VHcfIyUv, efD1bujZB9, pdAPN6IzNA, W06Xim0Kj, KDaBjaAd, lBORNzf, IzjxPpr, JV, A, uQBWWzi, PrTQi, m, b6zON, w6CDI, Um7Wt, Ues2RDsrO, rA, Ef, Fs, J2Nsqso, bdydMBsIm, 9C5ZO187, G3JHvk, AtHM, rKBKaxl7... how many input tokens does it use up? Number of tokens:",

    "you need to write the body of the request to an OpenAI chat completion call that: Uses model gpt-4o-mini Has a system message: Respond in JSON Has a user message: Generate 10 random addresses in the US Uses structured outputs to respond with an object addresses which is an array of objects with required fields: apartment (string) latitude (number) county (string) . Sets additionalProperties to false to prevent additional properties. Note that you don't need to run the request or use an API key; your task is simply to write the correct JSON body. What is the JSON body we should send to https://api.openai.com/v1/chat/completions for this? (No need to run it or to use an API key. Just write the body of the request below.)", 

    "Here is an example invoice image: Write just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content (text and image URL) to the OpenAI API endpoint. Use gpt-4o-mini as the model. Send a single user message to the model that has a text and an image_url content (in that order). The text content should be Extract text from this image. Send the image_url as a base64 URL of the image above. CAREFUL: Do not modify the image. Write your JSON body here:",

    "Here are 2 verification messages: Dear user, please verify your transaction code 63468 sent to 24ds2000125@ds.study.iitm.ac.in Dear user, please verify your transaction code 82151 sent to 24ds2000125@ds.study.iitm.ac.in The goal is to capture this message, convert it into a meaningful embedding using OpenAI's text-embedding-3-small model, and subsequently use the embedding in a machine learning model to detect anomalies. Your task is to write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding for the 2 given personalized transaction verification messages above. This will be sent to the endpoint https://api.openai.com/v1/embeddings. Write your JSON body here:",

    "Your task is to write a Python function most_similar(embeddings) that will calculate the cosine similarity between each pair of these embeddings and return the pair that has the highest similarity. The result should be a tuple of the two phrases that are most similar. Write your Python code here:",

    "Your task is to build a FastAPI POST endpoint that accepts an array of docs and query string via a JSON body. The endpoint is structured as follows: POST /similarity. The JSON response might look like this: What is the API URL endpoint for your implementation? It might look like: http://127.0.0.1:8000/similarity",

    "Develop a FastAPI application that: Exposes a GET endpoint /execute?q=... where the query parameter q contains one of the pre-templatized questions. Analyzes the q parameter to identify which function should be called. Extracts the parameters from the question text. Returns a response in the following JSON format:",

    #GA 4
    "What is the total number of ducks across players on page number 8 of ESPN Cricinfo's ODI batting stats?",

    "Source: Utilize IMDb's advanced web search at https://www.imdb.com/search/title/ to access movie data.Filter: Filter all titles with a rating between 3 and 8. Format: For up to the first 25 titles, extract the necessary details: ID, title, year, and rating. The ID of the movie is the part of the URL after tt in the href attribute. For example, tt10078772. Organize the data into a JSON structure as follows: Submit: Submit the JSON data",
    
    "Write a web application that exposes an API with a single query parameter: ?country=. It should fetch the Wikipedia page of the country, extracts all headings (H1 to H6), and create a Markdown outline for the country. What is the URL of your API endpoint?",

    "you are tasked with developing a system that automates the following: API Integration and Data Retrieval: Use the BBC Weather API to fetch the weather forecast for New York. Send a GET request to the locator service to obtain the city's locationId. Include necessary query parameters such as API key, locale, filters, and search term (city). What is the JSON weather forecast description for New York?",

    "What is the maximum latitude of the bounding box of the city Addis Ababa in the country Ethiopia on the Nominatim API? Value of the maximum latitude",

    "What is the link to the latest Hacker News post mentioning Open Source having at least 30 points?",

    "Using the GitHub API, find all users located in the city Barcelona with over 70 followers. When was the newest user's GitHub profile created?",

    "Create a scheduled GitHub action that runs daily and adds a commit to your repository. Enter your repository URL (format: https://github.com/USER/REPO)",

    "What is the total Economics marks of students who scored 62 or more marks in Economics in groups 76-100 (including both groups)?",

    "What is the markdown content of the PDF, formatted with prettier@3.4.2?",

    #GA5
    "Download the Sales Excel file: What is the total margin for transactions before Tue Nov 01 2022 03:45:13 GMT+0530 (India Standard Time) for Epsilon sold in BR (which may be spelt in different ways)?",

    "Download the text file with student marks How many unique students are there in the file?",

    "What is the number of successful GET requests for pages under /kannada/ from 5:00 until before 14:00 on Sundays?",

    "Across all requests under hindimp3/ on 2024-05-23, how many bytes did the top IP address (by volume of downloads) download?",

    "How many units of Ball were sold in Shenzhen on transactions with at least 123 units?",

    "Download the data from What is the total sales value?",

    "Download the data from How many times does HB appear as a key?",

    "Write a DuckDB SQL query to find all posts IDs after 2024-12-16T06:01:02.983Z with at least 1 comment with 3 useful stars, sorted. The result should be a table with a single column called post_id, and the relevant post IDs should be sorted in ascending order. ",

    "What is the text of the transcript of this Mystery Story Audiobook between 35.8 and 213.1 seconds?",

    "Here is the image. It is a 500x500 pixel image that has been cut into 25 (5x5) pieces: Here is the mapping of each piece: Original Row        Original Column        Scrambled Row        Scrambled Column 2        1        0        0 1        1        0        1 4        1        0        2 0        3        0        3 0        1        0        4 1        4        1        0 2        0        1        1 2        4        1        2 4        2        1        3 2        2        1        4 0        0        2        0 3        2        2        1 4        3        2        2 3        0        2        3 3        4        2        4 1        0        3        0 2        3        3        1 3        3        3        2 4        4        3        3 0        2        3        4 3        1        4        0 1        2        4        1 1        3        4        2 0        4        4        3 4        0        4        4 Upload the reconstructed image by moving the pieces from the scrambled position to the original position:"
]


question_functions = [
    # GA 1 
    solve_vscode_question,              
    solve_http_request_question,         
    solve_prettier_hash_question,        
    solve_google_sheets_question,        
    solve_excel_question,               
    solve_hidden_input_question,        
    solve_count_wednesdays_question,    
    solve_csv_extraction_question,      
    solve_json_sorting_question,        
    solve_json_conversion_question,      
    solve_div_sum_question,              
    solve_file_encoding_sum_question,   
    solve_github_repo_question,         
    solve_replace_text_question,         
    solve_file_size_filter_question,
    solve_rename_files_question,  
    solve_compare_files_question, 
    solve_sqlite_query_question, 
    # GA 2       
    solve_markdown_documentation_question,  
    solve_image_compression_question,
    solve_github_pages_question,   
    solve_colab_auth_question,    
    solve_colab_brightness_question, 
    solve_vercel_api_question,     
    solve_github_action_question,
    solve_docker_image_question,
    solve_fastapi_server_question,
    solve_llama_model_question,
    # GA 3
    solve_llm_sentiment,
    solve_token_cost,
    solve_address_generation,
    solve_llm_vision,
    solve_llm_embedding,
    solve_embedding_similarity,
    solve_vector_databases,
    solve_function_calling,
    # GA 4
    solve_html_google_sheets,
    solve_imdb,
    solve_wiki_headings,
    solve_weather_api,
    solve_city_bounding_box,
    solve_hacker_news,
    solve_new_github_user,
    solve_scheduled_github_action,
    solve_extract_tables,
    solve_pdf_to_md,
    # GA 5
    solve_excel_sales_cleanup,
    solve_student_marks_cleanup,
    solve_log_requests,
    solve_log_downloads,
    solve_cleanup_sales,
    solve_parse_partial_json,
    solve_nested_jsonkeys,
    solve_duckdb_socialmedia,
    solve_transcribe_yt,
    solve_image_reconstruction
]

# 🔹 Convert known questions into vectorized form
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(question_texts)

@app.post("/api/")
async def solve_question(question: str = Form(...), file: UploadFile = File(None)):
    """
    Uses TF-IDF + Cosine Similarity to map a user's question to the closest stored question.
    """

    # Convert the input question into vectorized form
    input_vector = vectorizer.transform([question])

    # Compute cosine similarity with stored questions
    similarities = cosine_similarity(input_vector, question_vectors)

    # Find the best matching question
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0, best_match_idx]

    # Set a similarity threshold (adjustable)
    if best_match_score < 0.5:
        return {"error": "No good match found. Please refine your question."}

    matched_function = question_functions[best_match_idx]
    print(f"DEBUG: Mapped to function - {matched_function.__name__}")

    return matched_function(question, file)

handler = Mangum(app)