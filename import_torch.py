import torch
from transformers import AutoTokenizer

# Load the tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the PyTorch model
model_path = "path/to/your/model.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the chatbot function
def chatbot():
    print("Hello, I'm a chatbot. How can I assist you today?")
    while True:
        # Get user input
        user_input = input("You: ")

        # Generate response using the pre-trained model
        with torch.no_grad():
            input_ids = tokenizer.encode(user_input, return_tensors='pt')
            response = model.generate(input_ids)

        # Convert the response tensor to text
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)

        # Print the input and response
        print(f"You: {user_input}")
        print(f"Bot: {response_text}")

        # Exit the loop if the user inputs "exit"
        if user_input.lower() == "exit":
            break

# Run the chatbot
chatbot()
