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


class AudioAgent:
    """
    Encapsulated agent for answering questions about an audio file.

    This agent works in two stages:
    1. Transcribe the audio file using an ASR model.
    2. Pass the transcription and the user's question to a chat model
       to get the final answer.
       
    Usage:
        # Assumes HF_TOKEN is in your environment
        agent = AudioAgent()
        
        # Ask a question with an audio file
        question = "What ingredients are in this recipe?"
        audio_path = "/path/to/your/recipe.mp3"
        answer = agent.answer(question, audio_path)
        
        # Or use the callable shortcut
        answer = agent(question, audio_path)
    """

    def __init__(
        self,
        asr_model_id: Optional[str] = None,
        chat_model_id: Optional[str] = None,
        provider: Optional[str] = "novita",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the AudioAgent.

        Args:
            asr_model_id: The repo ID of the ASR (transcription) model.
            chat_model_id: The repo ID of the chat model.
            provider: The provider for the chat InferenceClient.
            temperature: The generation temperature for the chat model.
            api_key: The API key. Defaults to os.environ["HF_TOKEN"].
        """
        # Model for speech-to-text
        self.asr_model_id = asr_model_id or "openai/whisper-large-v3"
        
        # Model for answering the question based on the text
        # Using a standard, high-performance chat model.
        # You can change this to "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        # if you know it also performs well on text-only tasks.
        self.chat_model_id = chat_model_id or "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        
        self.temperature = temperature
        
        # Get API key from arg or environment
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the HF_TOKEN environment variable "
                "or pass it as 'api_key' to the AudioAgent."
            )

        # 1. Client for ASR (using default HF provider)
        # We pass the full model ID to use the standard HF inference API
        self.asr_client = InferenceClient(
            model=self.asr_model_id, 
            api_key=self.api_key
        )

        # 2. Client for Chat (using your specified provider)
        # We pass the provider, and specify the model in the create call
        self.chat_client = InferenceClient(
            provider=provider, 
            api_key=self.api_key
        )

    # ---- Public API ----

    def answer(self, question: str, audio_path: str) -> str:
        """
        Run the agent to answer a question about an audio file.

        Args:
            question: The text prompt to ask.
            audio_path: The local file path to the audio file (e.g., .mp3).

        Returns:
            The parsed "Final Answer" string from the chat model.
        """
        # 1. Transcribe the audio
        try:
            transcription = self._transcribe(audio_path)
        except FileNotFoundError:
            return f"Error: Audio file not found at {audio_path}"
        except Exception as e:
            return f"Error during audio transcription: {e}"

        # 2. Build the combined prompt for the chat model
        combined_prompt = (
            f"You will be given a user's question and the transcription of an audio file. "
            f"Your task is to answer the question based *only* on the transcription.\n\n"
            f"--- TRANSCRIPTION ---\n"
            f"{transcription}\n"
            f"--- END TRANSCRIPTION ---\n\n"
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

    def __call__(self, question: str, audio_path: str) -> str:
        """Callable shortcut for self.answer()."""
        return self.answer(question, audio_path)

    # ---- Internals ----

    def _transcribe(self, audio_path: str) -> str:
        """Transcribes the audio file to text using the ASR client."""
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"No file found at {audio_path}")
        
        try:
            # Read audio file as bytes
            audio_bytes = p.read_bytes()
            
            # Call ASR API
            # This task returns a dict, e.g., {'text': '...'}
            response = self.asr_client.automatic_speech_recognition(audio_bytes)
            
            if "text" in response:
                return response["text"]
            else:
                raise ValueError(f"ASR API response did not contain 'text': {response}")
        except Exception as e:
            # Re-raise as a more informative exception
            raise RuntimeError(f"Error during audio transcription: {e}") from e

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
       python audio_agent.py "What ingredients are listed?" "path/to/my_recipe.mp3"
    """
    parser = argparse.ArgumentParser(description="Class-based agent for audio questions.")
    parser.add_argument("question", type=str, help="Your question (quoted).")
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    args = parser.parse_args()

    try:
        audio_agent = AudioAgent()
        answer = audio_agent.answer(args.question, args.audio_path)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()