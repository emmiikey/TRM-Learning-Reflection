from web_agent import WebAgent
from utils.fetch_files import fetch_files_gaia_validation_hub
from image_agent import ImageAgent
from audio_agent import AudioAgent
from python_agent import PythonAgent
from excel_agent import ExcelAgent
from youtube_agent import YouTubeSubsAgent
from gaia_agent import GaiaAgent

# ## testing the web agent alone
# test_question_web = "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?"
# webagent_instance = WebAgent()
# webagent_answer = webagent_instance(test_question_web)

## testing the image agent
# test_question_image = "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation."
# file_name = "cca530fc-4052-43b2-b130-b30968d8aa44.png"
# img_path = fetch_files_gaia_validation_hub(file_name)
# print(img_path)
# print("------- Image Agent ------------------")
# imageagent_instance = ImageAgent()
# imageagent_answer = imageagent_instance(test_question_image, img_path)
# print(imageagent_answer)

## testing the audio agent
# test_question_audio = "Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3.\n\nIn your response, please only list the ingredients, not any measurements. So if the recipe calls for \"a pinch of salt\" or \"two cups of ripe strawberries\" the ingredients on the list would be \"salt\" and \"ripe strawberries\".\n\nPlease format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients."
# file_name = "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"
# video_path = fetch_files_gaia_validation_hub(file_name)
# print(video_path)

# audioagent_instance = AudioAgent()
# audioagent_answer = audioagent_instance(test_question_audio, video_path)
# print(audioagent_answer)


## Testing the python code execution agent
# test_question_python = "What is the final numeric output from the attached Python code?"
# file_name = "f918266a-b3e0-4914-865d-4faa564f1aef.py"
# script_path = fetch_files_gaia_validation_hub(file_name)
# print(script_path)
# pythonagent_instance = PythonAgent()
# pythonagent_answer = pythonagent_instance(script_path)
# print(pythonagent_answer)

## Testing the excel agent
# test_question_excel = "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places."
# file_name = "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
# excel_path = fetch_files_gaia_validation_hub(file_name)
# print(excel_path)
# excelagent_instance = ExcelAgent()
# excelagent_answer = excelagent_instance(test_question_excel, excel_path)
# print(excelagent_answer)

## Testing youtube agent
# youtube_agent = YouTubeSubsAgent()
# ans = youtube_agent(
#     "What does Teal'c say in response to the question \"Isn't that hot?\"",
#     "https://www.youtube.com/watch?v=1htKBjuUWec"
# )
# print(ans)


gaia_agent = GaiaAgent()

# q = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
# ans = gaia_agent(q, "")

# q = "Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec. What does Teal'c say in response to the question \"Isn't that hot?\""
# ans = gaia_agent(q, "")

# file_img = fetch_files_gaia_validation_hub("cca530fc-4052-43b2-b130-b30968d8aa44.png")
# q = "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation."
# ans = gaia_agent(q, file_img)

# file_audio = fetch_files_gaia_validation_hub("1f975693-876d-457b-a649-393859e79bf3.mp3")
# q = "Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order."
# ans = gaia_agent(q, file_audio)

# file_xlsx = fetch_files_gaia_validation_hub("7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx")
# q = "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places."
# ans = gaia_agent(q, file_xlsx)

# file_py = fetch_files_gaia_validation_hub("f918266a-b3e0-4914-865d-4faa564f1aef.py")
# q = "What is the final numeric output from the attached Python code?"
# ans = gaia_agent(q, file_py)

q = ".rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"
ans = gaia_agent(q, "")

print(ans)

# from huggingface_hub import whoami
# whoami()