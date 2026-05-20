# gaia_agent.py
#
# High-level GAIA orchestrator with LLM-based routing.
#
# Usage:
#   from gaia_agent import GaiaAgent
#   from utils.fetch_files import fetch_files_gaia_validation_hub
#
#   gaia_agent = GaiaAgent()
#
#   # Q without file
#   ans1 = gaia_agent(
#       "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
#       ""
#   )
#
#   # Q with file (image)
#   img_path = fetch_files_gaia_validation_hub("cca530fc-4052-43b2-b130-b30968d8aa44.png")
#   ans2 = gaia_agent(
#       "Review the chess position provided in the image. It is black's turn...",
#       img_path
#   )
#
#   # Q with YouTube link
#   ans3 = gaia_agent(
#       "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
#       ""
#   )
#
#   print(ans1, ans2, ans3)

from __future__ import annotations

import os
import re
import mimetypes
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from web_agent import WebAgent
from image_agent import ImageAgent
from audio_agent import AudioAgent
from python_agent import PythonAgent
from excel_agent import ExcelAgent
from youtube_agent import YouTubeSubsAgent  # YouTubeSubsAgent(question, video_url)

load_dotenv()

#########################################################
# 1. Routing model prompt / helper
#########################################################

ROUTER_SYSTEM_PROMPT = """You are a routing controller for a GAIA QA system.
Your job is to look at a task and decide which specialized tool should answer it.

TOOLS:
- "web": For questions that need factual info from the internet, Wikipedia, sports stats, award numbers, historical rosters, etc. Use this if the answer relies on public info not contained in an attached file or transcript.
- "youtube": For questions that refer to a YouTube video URL and ask about what was said/heard in that video. The YouTube agent only has access to the video's subtitles.
- "image": For questions that require analyzing an image file (ex: chessboard, screenshot, diagram). File extensions like .png/.jpg/.jpeg.
- "audio": For questions that require transcribing or extracting info from an audio file (ex: .mp3, .wav, "listen to the recording and list ingredients").
- "excel": For questions that require reading a spreadsheet (.xls, .xlsx, .csv) and doing numeric reasoning or lookup in that sheet.
- "python": For questions that say "What is the final numeric output from the attached Python code?" or otherwise require running code from a .py file.
- "internal_reasoning": For pure logic, math, text manipulation, categorization, set theory, etc. No web. No external file beyond the text in the question.

RESTRICTIONS AND NOTES:
1. If there is a YouTube link in the question (youtube.com or youtu.be) and the user is asking about what someone says or does in that video, choose "youtube".
2. If there is an attached file_path and it's an image, choose "image".
3. If there is an attached file_path and it's audio, choose "audio".
4. If there is an attached file_path and it's Excel/CSV, choose "excel".
5. If there is an attached file_path and it's Python code (.py), choose "python".
6. Otherwise, if answering requires up-to-date or external factual knowledge (e.g. Wikipedia, sports stats, award numbers, who played in some show, etc.), choose "web".
7. Otherwise choose "internal_reasoning".

OUTPUT FORMAT:
Return ONLY one of the following exact strings:
web
youtube
image
audio
excel
python
internal_reasoning
"""

INTERNAL_REASONING_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

class RoutingLLM:
    """
    A tiny wrapper around the HF InferenceClient that decides the tool name.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        provider: str = "novita",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id or "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.temperature = temperature

        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError("RoutingLLM: HF_TOKEN not found and api_key not provided.")

        self.client = InferenceClient(provider=provider, api_key=self.api_key)

    def route(self, question: str, file_path: Optional[str]) -> str:
        """
        Ask the LLM which tool to use.
        We also inject lightweight hints (file extension, presence of YT link),
        so the LLM has explicit signals.
        """
        file_path = file_path or ""
        ext = os.path.splitext(file_path)[1].lower() if file_path else ""
        mime, _ = mimetypes.guess_type(file_path) if file_path else (None, None)
        has_youtube = "youtube.com" in question.lower() or "youtu.be" in question.lower()

        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "file_path:\n"
            f"{file_path}\n"
            f"(extension='{ext}', mime='{mime}', has_youtube={has_youtube})\n\n"
            "Which tool should handle this? Remember: output ONLY one of:\n"
            "web | youtube | image | audio | excel | python | internal_reasoning"
        )

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
        )

        raw = completion.choices[0].message.content.strip().lower()

        # clean it to a known label
        return self._normalize_route_label(raw)

    def _normalize_route_label(self, raw: str) -> str:
        """
        Normalize whatever the model said into one of the 7 valid labels,
        with a safe fallback.
        """
        allowed = {
            "web",
            "youtube",
            "image",
            "audio",
            "excel",
            "python",
            "internal_reasoning",
        }

        # grab the first token-like word that matches any allowed label
        for word in re.findall(r"[a-z_]+", raw):
            if word in allowed:
                return word

        # fallback: default to "web" because GAIA often needs retrieval
        return "web"


#########################################################
# 2. GaiaAgent Orchestrator
#########################################################

class GaiaAgent:
    """
    High-level GAIA orchestrator with an LLM router.

    Init:
        gaia_agent = GaiaAgent()

    Call:
        ans = gaia_agent(question, file_path)

    where:
        - question: str
        - file_path: str or "" (path to local file if provided for that task)
    """

    def __init__(
        self,
        chat_model_id: Optional[str] = None,
        router_model_id: Optional[str] = None,
        provider: str = "novita",
        temperature: float = 0.2,
        router_temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        self.chat_model_id = chat_model_id or "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.temperature = temperature

        # shared key/client
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError("GaiaAgent: HF_TOKEN not found and api_key not provided.")

        # core reasoning client (for internal_reasoning branch)
        self.reasoning_client = InferenceClient(
            provider=provider,
            api_key=self.api_key,
        )

        # router (LLM-based intent classifier)
        self.router = RoutingLLM(
            model_id=router_model_id or self.chat_model_id,
            provider=provider,
            temperature=router_temperature,
            api_key=self.api_key,
        )

        # instantiate sub-agents once
        self.web_agent = WebAgent()  # expects question
        self.youtube_agent = YouTubeSubsAgent(
            api_key=self.api_key,
        )  # expects (question, video_url)
        self.image_agent = ImageAgent(
            api_key=self.api_key,
        )  # expects (question, file_path)
        self.audio_agent = AudioAgent(
            api_key=self.api_key,
        )  # expects (question, file_path)
        self.python_agent = PythonAgent()  # expects (file_path)
        self.excel_agent = ExcelAgent(
            api_key=self.api_key,
        )  # expects (question, file_path)

    def __call__(self, question: str, file_path: Optional[str] = "") -> str:
        return self.answer(question, file_path)

    def answer(self, question: str, file_path: Optional[str] = "") -> str:
        """
        1. Ask router LLM which tool to use.
        2. Dispatch to that tool.
        3. Return the tool's final answer.
        """
        route = self.router.route(question, file_path)

        if route == "youtube":
            print("[GaiaAgent] Routing to YouTube agent.")
            video_url = self._extract_youtube_url(question)
            return self.youtube_agent(question, video_url if video_url else "")

        if route == "image":
            print("[GaiaAgent] Routing to Image agent.")
            return self.image_agent(question, file_path)

        if route == "audio":
            print("[GaiaAgent] Routing to Audio agent.")
            return self.audio_agent(question, file_path)

        if route == "excel":
            print("[GaiaAgent] Routing to Excel agent.")
            return self.excel_agent(question, file_path)

        if route == "python":
            print("[GaiaAgent] Routing to Python agent.")
            return self._run_python_agent(file_path)

        if route == "web":
            print("[GaiaAgent] Routing to Web agent.")
            return self.web_agent(question)

        # internal_reasoning or fallback
        print("[GaiaAgent] Routing to Internal Reasoning.")
        return self._internal_reasoning(question)

    #########################################################
    # internal helpers
    #########################################################

    def _extract_youtube_url(self, text: str) -> Optional[str]:
        """
        Pull first YouTube URL from question.
        """
        m = re.search(r"(https?://[^\s]+youtu[^\s]+)", text)
        if m:
            return m.group(1).strip().rstrip(').,]')
        return None

    def _run_python_agent(self, file_path: str) -> str:
        """
        PythonAgent currently is called like python_agent_instance(file_path)
        and returns output as string.
        """
        return self.python_agent(file_path)

    def _internal_reasoning(self, question: str) -> str:
        """
        Use the same GAIA-style 'FINAL ANSWER:' contract,
        but do pure reasoning, no web.
        """
        reasoning_prompt = (
            "You will be given a question that can be answered using reasoning, "
            "math, logic, text manipulation, categorization, or other internal thinking. "
            "Do not use any outside knowledge beyond what a normal educated human would know; "
            "do not fabricate web data.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Remember to follow the FINAL ANSWER format exactly."
        )

        messages = [
            {"role": "system", "content": INTERNAL_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": reasoning_prompt},
        ]

        completion = self.reasoning_client.chat.completions.create(
            model=self.chat_model_id,
            messages=messages,
            temperature=self.temperature,
        )

        raw = completion.choices[0].message.content
        return self._parse_final_answer(raw)

    def _parse_final_answer(self, response: str) -> str:
        """
        Extract text after 'FINAL ANSWER:' to stay consistent with other agents.
        """
        _before, _marker, final_answer = response.rpartition("FINAL ANSWER: ")
        if _marker:
            return final_answer.strip()
        return response.strip()
