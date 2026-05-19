from smolagents import DuckDuckGoSearchTool
from smolagents import Tool
import random
from huggingface_hub import list_models


# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            models = list(list_models(author=author, sort="downloads", limit=1))

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"


# Initialize custom tools
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()


if __name__ == "__main__":
    try:
        print(search_tool("President of France"))
    except Exception as e:
        print(f"Search tool test failed: {e}")

    print(weather_info_tool("Paris"))
    print(hub_stats_tool("facebook"))
