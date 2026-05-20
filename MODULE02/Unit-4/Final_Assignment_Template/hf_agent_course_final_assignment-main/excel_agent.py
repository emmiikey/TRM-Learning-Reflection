from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Import pandas for Excel/CSV handling
try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' and 'openpyxl' libraries are required for ExcelAgent.")
    print("Please install them with: pip install pandas openpyxl")
    exit(1)


# Load environment variables (e.g., HF_TOKEN) from a .env file
load_dotenv()

# This is the system prompt you provided
AGENT_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""


class ExcelAgent:
    """
    Encapsulated agent for answering questions about an Excel file.

    This agent works in two stages:
    1. Read the Excel file and convert its content to a text (CSV) format.
    2. Pass the user's question and the text-based data to a chat model
       to get the final answer.
       
    Usage:
        # Assumes HF_TOKEN is in your environment
        agent = ExcelAgent()
        
        # Ask a question with an Excel file
        question = "What are the total sales?"
        excel_path = "/path/to/your/data.xlsx"
        answer = agent.answer(question, excel_path)
        
        # Or use the callable shortcut
        answer = agent(question, excel_path)
    """

    def __init__(
        self,
        chat_model_id: Optional[str] = None,
        provider: Optional[str] = "novita",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the ExcelAgent.

        Args:
            chat_model_id: The repo ID of the chat model.
            provider: The provider for the chat InferenceClient.
            temperature: The generation temperature for the chat model.
            api_key: The API key. Defaults to os.environ["HF_TOKEN"].
        """
        # We use the same powerful model as the ImageAgent
        self.chat_model_id = chat_model_id or "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.temperature = temperature
        
        # Get API key from arg or environment
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the HF_TOKEN environment variable "
                "or pass it as 'api_key' to the ExcelAgent."
            )

        # Client for Chat
        self.chat_client = InferenceClient(
            provider=provider, 
            api_key=self.api_key
        )

    # ---- Public API ----

    def answer(self, question: str, excel_path: str) -> str:
        """
        Run the agent to answer a question about an Excel file.

        Args:
            question: The text prompt to ask.
            excel_path: The local file path to the Excel file (.xlsx).

        Returns:
            The parsed "Final Answer" string from the chat model.
        """
        # 1. Read Excel and convert to text
        try:
            data_as_text = self._excel_to_text(excel_path)
        except FileNotFoundError:
            return f"Error: Excel file not found at {excel_path}"
        except Exception as e:
            return f"Error during Excel processing: {e}"

        # 2. Build the combined prompt for the chat model
        combined_prompt = (
            f"You will be given a user's question and the content of an Excel file. "
            f"The Excel data is represented in CSV format. "
            f"Your task is to answer the question based *only* on the provided data.\n\n"
            f"--- DATA (CSV format) ---\n"
            f"{data_as_text}\n"
            f"--- END DATA ---\n\n"
            f"--- QUESTION ---\n"
            f"{question}\n"
            f"--- END QUESTION ---"
        )
        
        # 3. Build the messages payload
        messages = self._build_chat_messages(combined_prompt)

        # 4. Call the Chat API
        try:
            completion = self.chat_client.chat.completions.create(
                model=self.chat_model_id,
                messages=messages,
                temperature=self.temperature,
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            return f"Error calling Chat API: {e}"

        # 5. Parse and return the final answer
        return self._parse_answer(response_text)

    def __call__(self, question: str, excel_path: str) -> str:
        """Callable shortcut for self.answer()."""
        return self.answer(question, excel_path)

    # ---- Internals ----

    def _excel_to_text(self, excel_path: str) -> str:
        """Reads an Excel file and converts it to a CSV string."""
        p = Path(excel_path)
        if not p.exists():
            raise FileNotFoundError(f"No file found at {excel_path}")
        
        try:
            # Read the Excel file
            # By default, reads the first sheet
            df = pd.read_excel(str(p))
            
            # Convert DataFrame to a CSV string
            # index=False to avoid writing row numbers
            csv_string = df.to_csv(index=False)
            
            return csv_string
        except Exception as e:
            # Re-raise as a more informative exception
            raise RuntimeError(f"Error reading/converting Excel file: {e}") from e

    def _build_chat_messages(self, combined_prompt: str) -> List[Dict[str, Any]]:
        """Constructs the message payload for the chat API."""
        return [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": combined_prompt,
            },
        ]

    def _parse_answer(self, response: str) -> str:
        """Extracts the text after 'FINAL ANSWER: '."""
        try:
            # Use rpartition for robustness, splits on the *last* occurrence
            _before, _marker, final_answer = response.rpartition("FINAL ANSWER: ")
            
            if _marker:
                return final_answer.strip()
            else:
                # If marker isn't found, return the whole response with a warning
                print("Warning: Could not find 'FINAL ANSWER:' marker. Returning full response.")
                return response.strip()

        except Exception as e:
            print(f"Error parsing response: {e}")
            return response.strip()


# ---- CLI Entrypoint ----
def main():
    """
    Allows running the agent from the command line.
    
    Example:
       python excel_agent.py "What are the total sales?" "path/to/my_data.xlsx"
    """
    parser = argparse.ArgumentParser(description="Class-based agent for Excel questions.")
    parser.add_argument("question", type=str, help="Your question (quoted).")
    parser.add_argument("excel_path", type=str, help="Path to the Excel file.")
    args = parser.parse_args()

    try:
        excel_agent = ExcelAgent()
        answer = excel_agent.answer(args.question, args.excel_path)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()