import base64
import io
from PIL import Image
import requests
import pytesseract
import os
import subprocess
from gtts import gTTS
from pathlib import Path
import openai as OpenAI
from pydub import AudioSegment
import ffmpeg
from openai import OpenAI
from dotenv import load_dotenv

#Before running, add a .jpg to the image_path in line 36

# OpenAI API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert the image to grayscale
        img = img.resize((300, 300))  # Resize the image to reduce its size
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Path to your image
current_dir = os.path.dirname(os.path.abspath(__file__))

#Add picture of your equation you want to solve
image_path = os.path.join(current_dir, 'equations', 'PICTURE OF YOUR EQUATION.jpg')

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Function to make a single API call
def make_api_call_image(prompt, image):
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")
    response_data = response.json()
    if 'choices' not in response_data or len(response_data['choices']) == 0:
        raise Exception(f"Unexpected response format: {response_data}")
    return response_data['choices'][0]['message']['content'].strip()

def make_api_call_code(prompt):
    payload = {
        "model": "gpt-4-code-interpreter",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,  # Greedy decoding: always choose the highest probability token
        "top_p": 1
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")
    response_data = response.json()
    if 'choices' not in response_data or len(response_data['choices']) == 0:
        raise Exception(f"Unexpected response format: {response_data}")
    return response_data['choices'][0]['message']['content'].strip()

def make_api_call(prompt):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,  # Greedy decoding: always choose the highest probability token
        "top_p": 1
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")
    response_data = response.json()
    if 'choices' not in response_data or len(response_data['choices']) == 0:
        raise Exception(f"Unexpected response format: {response_data}")
    return response_data['choices'][0]['message']['content'].strip()

# Prompt 1: Given the image provided, return exactly and only the equation with an unknown variable in plain text.
prompt1 = f"Given the image provided, return exactly and only the equation with an unknown variable in the \n"
"           form of a latex formula. If you cannot confidently decide on an image, \n"
"           return /'unable to read equation/': data:image/jpeg;base64,{base64_image}\n"
"           remember, it should exclusively be a latex formula in the output, nothing else. DO NOT ADD ANY TEXT INTRODUCING THE FORMULA"

response1 = make_api_call_image(prompt1, base64_image)
equation = response1

# Prompt 2: Using elementary techniques, solve the equation in a way that is simple for a 6th grader to understand.
prompt2 = f"Solve the equation '{equation}' using your python code interpreter and the python sympy library. Once you are done, explain the steps in a way a 6th grader could understand."
response2 = make_api_call(prompt2)
solution_steps = response2

# Prompt 3: Make an animation in manim in the style of this code but substitute in the values from the equation in the image.
prompt3 = (
    f"Create an animation in manim using this code structure but with the equation '{equation}':\n\n"
    "There will be two parts to your response. 1st, a manim script\n"
    "In this part of your response, include exclusively the code and comments necessary to run in manim.\n"
    "Don't add anything else in this part because this response will be added to a code editor and run directly.\n"
    "note that to write the word or, you need \\text{or} \n"
    "Explanations in the code should be detailed, moreso than in the example. For example, explain how to factor if the equation calls for that. \n"

    "After you finish writing the manim code you will need to create the second part of the response, you will add\n"
    "this breakpoint to separate the two parts: *breakpoint* \n"
    "Create a second part of the response that is the steps transcribed into script that directly matches your explanations for each step.\n"
    "Make sure you add this second part, it won't work unless you add text after the *breakpoint*\n"
    "This is an example of a script you should create. It should be simple and exclusively the text in sentences so a text to speech can read it.\n"
    "i.e. Start with the equation. Subtract 4 from each side. Factor the equation. Solve for x\n"
    "The following is now and example of the manim code:\n"

    "from manim import *\n\n"
    "class SolveEquation(Scene):\n"
    "    def construct(self):\n"
    "        # Initial equation\n"
    "        equation1 = MathTex(\"x^2 - 3x = 4\")\n"
    "        explanation1 = Text(\"Start with the equation\").scale(0.6).to_edge(DOWN)\n"
    "        \n"
    "        # Step 1: Move 4 to the left side\n"
    "        equation2 = MathTex(\"x^2 - 3x - 4 = 0\")\n"
    "        explanation2 = Text(\"Subtract 4 from each side\").scale(0.6).to_edge(DOWN)\n"
    "        \n"
    "        # Step 2: Factor the quadratic equation\n"
    "        equation3 = MathTex(\"(x - 4)(x + 1) = 0\")\n"
    "        explanation3 = Text(\"Factor the equation\").scale(0.6).to_edge(DOWN)\n"
    "        \n"
    "        # Step 3: Solve for x\n"
    "        equation4 = MathTex(\"x - 4 = 0 \\, \\\text{or} \\, x + 1 = 0\")\n"
    "        equation5 = MathTex(\"x = 4 \\, \\\text{or} \\, x = -1\")\n"
    "        explanation4 = Text(\"Solve for x\").scale(0.6).to_edge(DOWN)\n"
    "        explanation5 = Text(\"Solve for x\").scale(0.6).to_edge(DOWN)\n"
    "        \n"
    "        # Initially position equations centered\n"
    "        equations = VGroup(equation1, equation2, equation3, equation4)\n"
    "        explanations = VGroup(explanation1, explanation2, explanation3, explanation4)\n"
    "        \n"
    "        for i, (eq, expl) in enumerate(zip(equations, explanations)):\n"
    "            if i == 0:\n"
    "                self.play(Write(eq))\n"
    "                self.add(expl)\n"
    "            else:\n"
    "                self.play(ReplacementTransform(equations[i-1], eq))\n"
    "                self.remove(explanations[i-1])\n"
    "                self.add(expl)\n"
    "            self.wait(2)\n"
    "        \n"
    "        self.play(ReplacementTransform(equation4, equation5))\n"
    "        self.remove(explanation4)\n"
    "        self.wait(2)\n"
    "        self.remove(equation5)\n"
    "        #reset equations back to initial states"

    "        equation1 = MathTex(\"x^2 - 3x = 4\")\n"
    "        equation2 = MathTex(\"x^2 - 3x - 4 = 0\")\n"
    "        equation3 = MathTex(\"(x - 4)(x + 1) = 0\")\n"
    "        equation4 = MathTex(\"x - 4 = 0 \\, \\\text{or} \\, x + 1 = 0\")\n"
    "        equation5 = MathTex(\"x = 4 \\, \\\text{or} \\, x = -1\")\n"
    "        all_equations = [equation1, equation2, equation3, equation4, equation5]"

    "        # Move equations to the left side for the second part\n"
    "        self.play(VGroup(*all_equations).animate.arrange(DOWN).to_edge(LEFT))\n"
    "        \n"
    "        # Graphing\n"
    "        axes = Axes(\n"
    "            x_range=[-3, 5, 1],\n"
    "            y_range=[-5, 10, 1],\n"
    "            axis_config={\"include_tip\": True}\n"
    "        ).scale(0.5).shift(RIGHT * 3)\n"
    "        \n"
    "        graph = axes.plot(lambda x: x**2 - 3*x - 4, x_range=[-3, 5], color=BLUE)\n"
    "        solution_dot1 = Dot(axes.coords_to_point(4, 0), color=RED)\n"
    "        solution_label1 = MathTex(\"(4, 0)\").next_to(solution_dot1, UP)\n"
    "        solution_dot2 = Dot(axes.coords_to_point(-1, 0), color=RED)\n"
    "        solution_label2 = MathTex(\"(-1, 0)\").next_to(solution_dot2, UP)\n"
    "        \n"
    "        self.play(Write(axes))\n"
    "        self.play(Create(graph))\n"
    "        self.play(Create(solution_dot1), Write(solution_label1))\n"
    "        self.play(Create(solution_dot2), Write(solution_label2))\n"
    "        self.wait(4)\n"
)

response3 = make_api_call(prompt3)
manim_code = response3

# Print the results
print(f"Prompt 1: {prompt1, base64_image}")
print(f"Response 1: {equation}")
print('-' * 80)
print(f"Prompt 2: {prompt2}")
print(f"Response 2: {solution_steps}")
print('-' * 80)
print(f"Prompt 3: {prompt3}")
print(f"Response 3: {manim_code}")
print('-' * 80)

def clean_manim_code(raw_response):
    # Isolate the script up until *breakpoint*
    breakpoint_index = raw_response.find('*breakpoint*')
    if breakpoint_index != -1:
        code_segment = raw_response[:breakpoint_index].strip()
    else:
        code_segment = raw_response.strip()

    # Remove the ```python and ``` backticks
    cleaned_code = code_segment.replace("```python", "").replace("```", "").strip()
    return cleaned_code

def clean_description(raw_response):
    # Isolate everything after *breakpoint*
    breakpoint_index = raw_response.find('*breakpoint*')
    if breakpoint_index != -1:
        description_segment = raw_response[breakpoint_index + len('*breakpoint*'):].strip()
    else:
        description_segment = raw_response.strip()

    # Remove the ``` backticks
    cleaned_description = description_segment.replace("```", "").strip()
    return cleaned_description

def text_to_sentences(text):
    # Split the text into sentences
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    # Add a period to each sentence if needed
    sentences = [sentence + '.' if not sentence.endswith('.') else sentence for sentence in sentences]
    return sentences

def save_manim_code(manim_code):
    with open("solve_equation.py", "w") as file:
        file.write(manim_code)

def run_manim_script():
    command = ["manim", "-pql", "solve_equation.py", "SolveEquation"]
    subprocess.run(command)

narration = clean_description(manim_code)
sentence_list = text_to_sentences(narration)
print(narration)

manim_code = clean_manim_code(manim_code)
save_manim_code(manim_code)
print(manim_code)

def create_speech_with_intervals(sentences, output_file="combined_speech.mp3", total_duration=3.5):
    # Directory to save temporary audio files
    audio_dir = Path(__file__).parent / "temp_audio_files"
    audio_dir.mkdir(exist_ok=True)

    # Function to create speech from text using OpenAI API
    def create_speech(text, filename):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize the client
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        speech_file_path = Path(filename)
        response.stream_to_file(speech_file_path)

    # Create each audio file and save it
    audio_files = []
    for i, sentence in enumerate(sentences):
        audio_file = audio_dir / f"sentence_{i+1}.mp3"
        create_speech(sentence, audio_file)
        audio_files.append(audio_file)

    # Combine all audio files with intervals
    combined_audio = AudioSegment.empty()
    for i, audio_file in enumerate(audio_files):
        sentence_audio = AudioSegment.from_mp3(audio_file)
        silence_duration = max(total_duration * 1000 - len(sentence_audio), 0)  # Calculate silence to add
        silence = AudioSegment.silent(duration=silence_duration)
        combined_audio += sentence_audio + silence

    # Export the combined audio to the output file
    combined_audio.export(output_file, format="mp3")

    # Cleanup: Remove temporary audio files
    for audio_file in audio_files:
        os.remove(audio_file)
    os.rmdir(audio_dir)

create_speech_with_intervals(sentence_list)

def combine_video_audio(video_file, audio_file, output_file):
    input_video = ffmpeg.input(video_file)
    input_audio = ffmpeg.input(audio_file)

    ffmpeg.output(input_video, input_audio, output_file, vcodec='copy', acodec='aac').run()

run_manim_script()

video_file = Path('media/videos/solve_equation/480p15/SolveEquation.mp4')
audio_file = Path('combined_speech.mp3')

combine_video_audio(video_file, audio_file, 'final_output2.mp4')


