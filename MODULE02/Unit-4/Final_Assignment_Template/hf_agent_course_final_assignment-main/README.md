---
title: Agent Course Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Question analysis

We need following functionalities in our agent:
- [x] General orchestrator agent: base on the question, recognize what tools are needed to answer the question, call the tools, collect the results, and generate the final answer.
- [x] youtube video analysis
- [x] excel manipulation
- [x] web search (duckduckgo react agent)
- [x] within web search, we need also to search published papers
- [x] python exection (done, without llm, only naive code execution)
- [x] audio analysis (api calling inference client of mp3 transcript + chat model)
- [x] image analysis (api calling inference client of vlm model)