from smolagents import CodeAgent, InferenceClientModel
from retriever import load_guest_dataset
from tools import search_tool, weather_info_tool, hub_stats_tool

# Load the guest dataset and retriever tool
guest_info_tool = load_guest_dataset()

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred with the guest info tool
alfred = CodeAgent(
    tools=[
        guest_info_tool,
        weather_info_tool,
        hub_stats_tool,
        search_tool
    ],
    model=model,
    add_base_tools=True,
    planning_interval=3
)

# Example query
response = alfred.run("I need to speak with Dr. Nikola Tesla from the guest list. Can you help me prepare for this conversation?")

print("Alfred's Response:")
print(response)