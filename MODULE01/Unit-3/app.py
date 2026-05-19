from smolagents import CodeAgent, InferenceClientModel
from retriever import load_guest_dataset

# Load the guest dataset and retriever tool
guest_info_tool = load_guest_dataset()

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred with the guest info tool
alfred = CodeAgent(
    tools=[guest_info_tool],
    model=model
)

# Example query
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("Alfred's Response:")
print(response)