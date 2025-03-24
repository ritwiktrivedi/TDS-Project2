from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import zipfile
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Callable
import io
import base64

app = FastAPI()

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionInfo:
    def __init__(self, id:str, keywords:List[str], solver: callable):
        self.id = id
        self.keywords = keywords
        self.solver = solver

@app.get("/api/test")
async def test():
    return {"message": "Hello from FastAPI on Vercel"}

def extract_files(file:UploadFile) -> Dict[str, Any]:
    """Extract and process uploaded files"""
    content = {}
    file_content = file.file.read()

    if file.filename.endswith('.zip'):
        with zipfile.ZipFile(BytesIO(file_content)) as z:
            for filename in z.namelist():
                if filename.endswith('.csv'):
                    with z.open(filename) as f:
                        content[filename] = pd.read_csv(f)
                elif filename.endswith(('.xls', '.xlsx')):
                    with z.open(filename) as f:
                        content[filename] = pd.read_excel(f)
                elif filename.endswith('.json'):
                    with z.open(filename) as f:
                        content[filename] = json.load(f)
    elif file.filename.endswith('.csv'):
        content[file.filename] = pd.read_csv(BytesIO(file_content))
    elif file.filename.endswith(('.xls','.xlsx')):
        content[file.filename] = pd.read_excel(BytesIO(file_content))
    elif file.filename.endswith('.json'):
        content[file.filename] = json.loads(file_content.decode('utf-8'))
    else:
        try:
            content[file.filename] = json.loads(file_content.decode('utf-8'))
        except:
            content[file.filename] = file_content

    return content


def identify_question(question_text: str) -> List[str]:
    """identify question based on keyword"""
    question_text = question_text.lower()
    identified_questions = set()

    for q_id, q_info in QUESTIONS.items():
        for keyword in q_info.keywords:
            if keyword.lower() in question_text:
                identified_questions.add(q_id)
                break

    return list(identified_questions) if identified_questions else []

# Question solver functions
def solve_descriptive_stats(data_files: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Calculation for basic descriptive stats"""
    results = {}

    for filename, df in data_files.items():
        if isinstance(df, pd.DataFrame):
            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if num_cols:
                stats = {}
                for col in num_cols:
                    stats[col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                    }
                results[filename] = stats
    
    return {
        'type': 'descriptive_statistics',
        'results': results
    }

def solve_json_sort(data_files: Dict[str, Any], question:str) -> Dict[str, Any]:
    """sort json based on specific fields"""
    results = {}

    # If the question has JSON directly in it, extract and sort it
    if "[(" in question and ")]" in question:
        try:
            # Find JSON array in the question
            start_idx = question.find("[{")
            end_idx = question.find("}]") + 2
            json_str = question[start_idx:end_idx]

            # Parse the JSON array
            data = json.loads(json_str)

            # Sort data by age first, then by name for ties
            sorted_data = sorted(data, key=lambda x: (x.get("age", 0), x.get("name", "")))

            # Create the JSON string with no extra spaces or newlines
            compact_json = json.dumps(sorted_data, separators=(",", ":"))

            results["json_from_question"] = {
                    "sorted_data": sorted_data,
                    "compact_json": compact_json
                }
            

            return {
                'type': 'json_sort',
                'results': results
            }

        except Exception as e:
            return {
                'type': 'error',
                'message': f'Error processing JSON: {str(e)}'
            }

    for filename, content in data_files.items():
        if isinstance(content, list):
            sorted_data = sorted(
                content,
                key=lambda x: (x.get("age", 0), x.get("name",""))
                
            )

            compact_json = json.dumps(sorted_data, separators=(',', ':'))

            results[filename] = {
                "sorted_data": sorted_data,
                "compact_json": compact_json
            }

    return {
        'type': "json_sort",
        'results': results
    }

QUESTIONS = {
    'q1': QuestionInfo(
        id ='q1',
        keywords=['mean','median','average', 'descriptive statistics', 'summary statistics'],
        solver=solve_descriptive_stats
    ),
    'q4': QuestionInfo(
        id='q4',
        keywords=['sort', 'json', 'array','objects','ascending', 'descending', 'order'],
        solver=solve_json_sort
    )
}

@app.post("/api")
@app.post("/api/")
async def process_request(question: str = Form(...), file:UploadFile = Form(...)):
    try:
        file_content = extract_files(file)

        if "sort this JSON array" in question and "[{" in question and "}]" in question:
            result = solve_json_sort(file_content, question)
            return {
                "status": "succes",
                "identified_question": "q4",
                "message": "Sorted JSON array",
                "result": result
            }
        
        question_ids = identify_question(question)

        if not question_ids:
            return {
                "status": "error",
                "message": "no matching question identified"
            }
        
        question_id = question_ids[0]
        question_info = QUESTIONS.get(question_id)

        if question_info and question_info.solver:
            result = question_info.solver(file_content, question)

            return {
                "status": "success",
                "identified_question" : question_id,
                "message" : f"Answered question related to: {question_id}",
                "result": result
            }
        else:
            return {
                "status": "error",
                "message": f"question identified ({question_id}), but no solver available yet"
            }

    except Exception as e:
        return {"status": "error", "message": str(e), "details":str(type(e))}