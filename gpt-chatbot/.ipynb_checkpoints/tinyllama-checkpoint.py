from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_response(prompt, max_length=300, num_return_sequences=1):
    # Tokenize and encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Move input_ids to the same device as the model
    input_ids = input_ids.to(device)

    # Move model to the same device as input_ids
    model.to(input_ids.device)

    # Generate response
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2)

    # Decode and return the generated response
    generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_response


# Usage
problem = "Businesses worldwide expend substantial financial resources on paper-based transaction evidence like printed receipts. This not only adds to operational costs but also contributes to environmental degradation due to paper wastage and lack of recycling."
prompt = "Propose a solution for this problem: " + problem
response = generate_response(prompt)
print("Generated Response:", response)
