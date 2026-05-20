# web_agent.py
# Minimal web-searching LangChain agent for GAIA-style questions.
# - Tools: DuckDuckGo search + HTTP GET
# - Model: defaults to a lightweight reasoning-ready open model (Hugging Face), but you can
#          swap to any chat model supported by LangChain. If you don't set HF token, we'll
#          fall back to a simple reasoning prompt with the same interface.

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Dict, Any

# LangChain core (v0.2+ / v0.3)
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities import RequestsWrapper
from langchain.tools import Tool

# Chat model: Hugging Face Endpoint (no key required to run, but highly recommended)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Community tools
from langchain_community.tools import DuckDuckGoSearchRun, RequestsGetTool

# ---- Prompt (self-contained: no hub dependency) ----
REACT_PROMPT = """You are a meticulous research assistant for GAIA-style fact questions.

You have access to these tools:
{tools}

Use this format:

Question: the input question to answer
Thought: think about what to search or read next
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I can now answer.
Final Answer: <the shortest exact answer>

Follow the rules:
- Prefer precise quotes from primary/official sources when possible.
- Keep the final answer SHORT (a few words if possible)
- If you need multiple pages, open them sequentially.
- Never repeat the exact same Action Input twice. If an Action returns a login, auth, or error page, change strategy (use search with site: filters, try a different public page, or a cached/mirrored source).
- If a URL contains "login", "@app/auth", or "returnto=", do NOT open it; search for a public page instead.

Question: {input}
{agent_scratchpad}"""

# Final Answer: <the shortest exact answer> — <single best source URL>
# - Keep the final answer SHORT (a few words if possible) plus one URL.
class WebAgent:
    """Encapsulated GAIA-style web agent.

    Usage:
        web_agent = WebAgent()
        result = web_agent("Who is the CEO of ...?")
        # or
        result = web_agent.answer("...")

    You can also run as a script:
        python web_agent.py "Your question here"
    """

    def __init__(
        self,
        hf_repo_id: Optional[str] = None,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        max_iterations: int = 20,
        verbose: bool = True,
        user_agent: str = "Mozilla/5.0 (compatible; GAIA/1.0; +https://example.org/bot)",
    ):
        self.hf_repo_id = hf_repo_id or os.environ.get("HF_REPO_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.user_agent = user_agent

        # Built lazily so each call gets a fresh per-run "seen URL" set for the guarded GET.
        self._executor: Optional[AgentExecutor] = None
        self._tools: Optional[List[Tool]] = None

    # ---- Public API ----
    def answer(self, question: str) -> str:
        """Run the agent and return the model's 'Final Answer' string."""
        executor = self._build_agent()  # build fresh each time to reset guarded state
        result: Dict[str, Any] = executor.invoke({"input": question})
        return result["output"]

    def __call__(self, question: str) -> str:
        return self.answer(question)

    # ---- Internals ----
    def _build_model(self) -> ChatHuggingFace:
        llm = HuggingFaceEndpoint(
            repo_id=self.hf_repo_id,
            task="text-generation",
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
        )
        return ChatHuggingFace(llm=llm)

    def _build_tools(self) -> List[Tool]:
        wrapper = RequestsWrapper(headers={"User-Agent": self.user_agent})
        raw_get = RequestsGetTool(
            name="Requests Get (raw)",
            requests_wrapper=wrapper,
            allow_dangerous_requests=True,
        )

        seen: set[str] = set()

        def guarded_get(url: str) -> str:
            low = url.lower()
            if any(x in low for x in ["@app/auth", "login?returnto=", "/login"]):
                return ("BLOCKED: This looks like a login/auth wall. "
                        "Use search to find a public or cached page instead.")
            if url in seen:
                return (f"BLOCKED: Already fetched {url}. "
                        "Choose a different URL or try a different query.")
            seen.add(url)
            return raw_get.run(url)

        return [
            DuckDuckGoSearchRun(name="DuckDuckGo Search"),
            Tool.from_function(
                name="Requests Get",
                func=guarded_get,
                description="HTTP GET for public pages; prevents repeats and login walls.",
            ),
        ]

    def _build_agent(self) -> AgentExecutor:
        model = self._build_model()
        tools = self._build_tools()
        prompt = PromptTemplate.from_template(REACT_PROMPT)

        agent = create_react_agent(model, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            verbose=self.verbose,
        )
        return executor


# def main():
#     parser = argparse.ArgumentParser(description="Simple LangChain web agent for GAIA-style questions.")
#     parser.add_argument("question", type=str, help="Your question (quoted).")
#     args = parser.parse_args()

#     print(answer(args.question))


# if __name__ == "__main__":
#     main()

# ---- CLI Entrypoint ----
def main():
    parser = argparse.ArgumentParser(description="Class-based LangChain web agent for GAIA-style questions.")
    parser.add_argument("question", type=str, help="Your question (quoted).")
    args = parser.parse_args()

    web_agent = WebAgent()
    print(web_agent.answer(args.question))


if __name__ == "__main__":
    main()
