import huggingface

# Set up Hugging Face API credentials
huggingface.api_key = "YOUR_Hugging_Face_API_KEY"

# Create a new chatbot
chatbot = ""

# Provide examples to train GPT-3 model
training_data = [
    ("book_hotel", "I want to book a hotel in New York City"),
    ("get_weather", "The weather in New York City is sunny today")
]

# Prepare training prompts and completions
prompts = [{"role": "system", "content": "You are a chatbot that helps with hotel booking and weather queries."}]
for intent, example in training_data:
    prompts.append({"role": "user", "content": example})
    prompts.append({"role": "assistant", "content": intent})

# Train the chatbot
chatbot = huggingface.Completion.create(
    engine="text-davinci-003",
    prompt=prompts,
    temperature=0.8,
    max_tokens=100
)

# Test the chatbot by querying for weather
query = "What's the weather like?"
response = huggingface.Completion.create(
    engine="text-davinci-003",
    prompt=[{"role": "user", "content": query}] + chatbot.choices,
    temperature=0.8,
    max_tokens=50
)
weather_output = response.choices[0].text.strip()

print(weather_output)  # Output: The weather in New York City is sunny today

# Test the chatbot by querying for hotel booking
query = "I want to book a hotel."
response = huggingface.Completion.create(
    engine="text-davinci-003",
    prompt=[{"role": "user", "content": query}] + chatbot.choices,
    temperature=0.8,
    max_tokens=50
)
hotel_output = response.choices[0].text.strip()

print(hotel_output)  # Output: I can help you with that. Please provide more information.
