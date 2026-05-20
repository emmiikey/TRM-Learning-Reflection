# youtube_subs_agent.py
# GAIA-style YouTube subtitle-based agent using youtube-transcript-api.
#
# Usage:
#   pip install youtube-transcript-api python-dotenv huggingface_hub
#
# Example:
#   from youtube_subs_agent import YouTubeSubsAgent
#
#   agent = YouTubeSubsAgent()
#   answer = agent(
#       "What does Teal'c say in response to the question 'Isn't that hot?'",
#       "https://www.youtube.com/watch?v=1htKBjuUWec"
#   )
#   print(answer)

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from huggingface_hub import InferenceClient

load_dotenv()

AGENT_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""


class YouTubeSubsAgent:
    """
    GAIA-style agent for answering questions about YouTube videos using subtitles.

    Usage:
        agent = YouTubeSubsAgent()
        answer = agent(
            "What does Teal'c say in response to the question 'Isn't that hot?'",
            "https://www.youtube.com/watch?v=1htKBjuUWec"
        )
    """

    def __init__(
        self,
        chat_model_id: Optional[str] = None,
        provider: str = "novita",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        transcript_char_limit: int = 12000,
        languages: Optional[List[str]] = None,
    ):
        """
        languages: priority list for transcript languages.
                   default = English-first fallback.
        """
        self.chat_model_id = chat_model_id or "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.temperature = temperature
        self.transcript_char_limit = transcript_char_limit
        self.languages = languages or ["en", "en-US", "en-GB"]

        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the HF_TOKEN environment variable or pass api_key=."
            )

        self.client = InferenceClient(provider=provider, api_key=self.api_key)

    # ---- Public API ----

    def answer(self, question: str, video_url: str) -> str:
        """Main call method: question + YouTube URL."""
        transcript_text = self._fetch_transcript_text(video_url)
        prompt = self._build_reasoning_prompt(question, transcript_text)
        response = self._run_chat(prompt)
        return self._parse_final_answer(response)

    def __call__(self, question: str, video_url: str) -> str:
        """Shortcut for answer()."""
        return self.answer(question, video_url)

    # ---- Internals ----

    def _extract_video_id(self, url: str) -> str:
        """
        Extracts the video ID from common YouTube URL formats.
        Falls back to returning the input if it looks like a bare ID already.
        """
        parsed = urlparse(url)

        # Short link: https://youtu.be/VIDEOID
        if parsed.hostname in ("youtu.be",):
            # path is like "/VIDEOID"
            return parsed.path.strip("/")

        # Standard watch link: https://www.youtube.com/watch?v=VIDEOID
        if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]

        # Fallback: assume caller already passed just the ID
        return url

    def _fetch_transcript_text(self, video_url: str) -> str:
        """
        Fetch transcript with youtube-transcript-api, format it as
        MM:SS text
        one line per snippet, and truncate if needed.
        """
        video_id = self._extract_video_id(video_url)

        try:
            fetched = YouTubeTranscriptApi().fetch(
                video_id,
                languages=self.languages,
            )
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            print(f"Warning: no transcript available for this video ({e}).")
            return ""

        # IMPORTANT FIX:
        # 'fetched' is a FetchedTranscript, which is iterable, but each element
        # is a FetchedTranscriptSnippet (attributes, not dicts).
        # Easiest: convert to raw dicts first.
        raw_snippets = fetched.to_raw_data()

        def format_ts(seconds_float: float) -> str:
            seconds_int = int(seconds_float)
            mm = seconds_int // 60
            ss = seconds_int % 60
            return f"{mm:02d}:{ss:02d}"

        lines: List[str] = []
        for snip in raw_snippets:
            start_ts = format_ts(snip["start"])
            text = snip["text"]
            # Collapse internal newlines that YouTube sometimes puts in a single snippet
            text = text.replace("\n", " ").strip()
            if text:
                lines.append(f"{start_ts} {text}")

        transcript_text = "\n".join(lines)

        if len(transcript_text) > self.transcript_char_limit:
            transcript_text = transcript_text[: self.transcript_char_limit] + "\n[truncated]"

        return transcript_text

    def _build_reasoning_prompt(self, question: str, transcript_text: str) -> str:
        """
        Create the reasoning prompt for the LLM.
        """
        return (
            "You will be given a question about a YouTube video's dialogue and its transcript.\n"
            "Answer ONLY using what is said in the transcript. If the user asks for an exact quote, "
            "repeat the line(s) exactly as spoken.\n\n"
            "----- QUESTION -----\n"
            f"{question}\n"
            "----- END QUESTION -----\n\n"
            "----- TRANSCRIPT -----\n"
            f"{transcript_text if transcript_text else '[no transcript available]'}\n"
            "----- END TRANSCRIPT -----\n"
        )

    def _run_chat(self, prompt: str) -> str:
        """
        Send the reasoning prompt to the chat model.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.chat_model_id,
            messages=messages,
            temperature=self.temperature,
        )

        return completion.choices[0].message.content

    def _parse_final_answer(self, response: str) -> str:
        """
        Extract 'FINAL ANSWER: ...' just like in your other agents.
        """
        _before, _marker, final_answer = response.rpartition("FINAL ANSWER: ")
        if _marker:
            return final_answer.strip()
        else:
            return response.strip()

# ---- CLI Entrypoint ----
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GAIA-style YouTube subtitles agent (youtube-transcript-api).")
    parser.add_argument("question", type=str, help='Your question (quoted).')
    parser.add_argument("video_url", type=str, help="YouTube video URL.")
    args = parser.parse_args()

    try:
        agent = YouTubeSubsAgent()
        answer = agent(args.question, args.video_url)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()