from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the pre-trained model
model = torch.load("model.pt")

# Load the pre-trained model
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model)

# Set the model to evaluation mode
model.eval()

# Define the interface function
def generate_output(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output using the pre-trained model
    with torch.no_grad():
        output = model.generate(input_ids)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text
    return generated_text
