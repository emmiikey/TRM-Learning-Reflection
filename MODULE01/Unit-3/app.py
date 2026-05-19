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
        search_tool,
        weather_info_tool,
        hub_stats_tool
    ],
    model=model
)

# Example query
response = alfred.run("What is the weather in Paris and what is Facebook's most downloaded Hugging Face model?")

print("Alfred's Response:")
print(response)