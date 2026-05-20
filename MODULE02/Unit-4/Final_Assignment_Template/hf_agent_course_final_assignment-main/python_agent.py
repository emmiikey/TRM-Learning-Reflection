from __future__ import annotations

import os
import base64
import mimetypes
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import io
from contextlib import redirect_stdout

load_dotenv()

class PythonAgent:
    def __init__(self):
        # print("Python code execution agent initialized.")
        pass
    
    def __call__(self, file_path: str) -> str:
        return self.execute_python_file(file_path)


    def execute_python_file(self, file_path: str) -> str:
        import io
        from contextlib import redirect_stdout

        with open(file_path, "r") as f_in:
            code_to_exec = f_in.read()

        # important: real module-like globals with __name__="__main__"
        sandbox_globals = {
            "__name__": "__main__",
            "__file__": file_path,
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                exec(code_to_exec, sandbox_globals)
            except Exception as e:
                print(f"An error occurred: {e}")

        raw_response = buf.getvalue()
        # the response is the last line
        response = raw_response.strip().splitlines()[-1]
        return response

# path_python_file = r"downloaded\f918266a-b3e0-4914-865d-4faa564f1aef.py"
# response = execute_python_file(path_python_file)
# print(response)
