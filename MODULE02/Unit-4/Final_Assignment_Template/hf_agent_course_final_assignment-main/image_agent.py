from __future__ import annotations

import os
import base64
import mimetypes
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables (e.g., HF_TOKEN) from a .env file
load_dotenv()

# This is the system prompt you provided
AGENT_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""


class ImageAgent:
    """
    Encapsulated agent for answering questions about an image.

    Usage:
        # Assumes HF_TOKEN is in your environment
        agent = ImageAgent()
        
        # Ask a question with an image
        question = "What is in this image?"
        image_path = "/path/to/your/image.png"
        answer = agent.answer(question, image_path)
        
        # Or use the callable shortcut
        answer = agent(question, image_path)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        provider: Optional[str] = "novita",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the ImageAgent.

        Args:
            model_id: The repository ID of the model to use.
            provider: The provider for the InferenceClient.
            temperature: The generation temperature.
            api_key: The API key. Defaults to os.environ["HF_TOKEN"].
        """
        self.model_id = model_id or "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.temperature = temperature
        
        # Get API key from arg or environment, raising an error if missing
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the HF_TOKEN environment variable "
                "or pass it as 'api_key' to the ImageAgent."
            )

        # Initialize the client
        self.client = InferenceClient(
            provider=provider,
            api_key=self.api_key,
        )

    # ---- Public API ----

    def answer(self, question: str, image_path: str) -> str:
        """
        Run the agent to answer a question about an image.

        Args:
            question: The text prompt to ask the model.
            image_path: The local file path to the image.

        Returns:
            The parsed "Final Answer" string from the model.
        """
        # 1. Convert image to data URL
        try:
            image_data_url = self._to_data_url(image_path)
        except FileNotFoundError:
            return f"Error: Image file not found at {image_path}"
        except Exception as e:
            return f"Error processing image: {e}"

        # 2. Build the messages payload
        messages = self._build_messages(question, image_data_url)

        # 3. Call the API
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            return f"Error calling API: {e}"

        # 4. Parse and return the final answer
        return self._parse_answer(response_text)

    def __call__(self, question: str, image_path: str) -> str:
        """Callable shortcut for self.answer()."""
        return self.answer(question, image_path)

    # ---- Internals ----

    def _to_data_url(self, path: str) -> str:
        """Converts a local file path to a base64 data URL."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No file found at {path}")
            
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _build_messages(self, question: str, image_data_url: str) -> List[Dict[str, Any]]:
        """Constructs the message payload for the API."""
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": AGENT_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
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
       python image_agent.py "What move is this?" "path/to/my_image.png"
    """
    parser = argparse.ArgumentParser(description="Class-based agent for image questions.")
    parser.add_argument("question", type=str, help="Your question (quoted).")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    try:
        image_agent = ImageAgent()
        answer = image_agent.answer(args.question, args.image_path)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()