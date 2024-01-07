import streamlit as st
import torch
from transformers import pipeline

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map=device)

def main():
    st.title("Solution Evaluator")

    # Input box for user to enter messages
    problem_input = st.text_input("Enter the Problem you want to solve:", "")
    solution_input = st.text_input("Enter the Solution to evaluate:", "")

    if problem_input != "" and solution_input != "" :
    # Display user and bot messages
        generate_report(problem_input, solution_input)


def generate_report(problem_input, solution_input):

    messages = [
        # {
        #     "role": "system",
        #     "content": """You are a report generating chatbot that evaluates the quality of a solution to a problem. For any input
        #     you will generate an evaluation report in the following format:  
        #     - Relevance: [A number on a scale of 1- 5 indicating the relevance of the solution to the problem] out of 5
        #     - Circular Economy Fit: [A number on a scale or 1-5 indicating fit into at least one of the three principles:
        #         1) Waste and Pollution Elimination,
        #         2) Product Circulation,
        #         3) Nature Regeneration] out of 5
        #     - Market Potential: [A number on a scale of 1-5 indicating Market size and saturation, Current demands, Growth potentials] / 5
        #     - Feasibility: [A number on a scale of 1-5 indicating Implementation Difficulties and Risks] out of 5
        #     - Overall Rating: [Average of Relevance, Circular Economy Fit, Market Potential, Feasibility scores] out of 5
        #     - Improvement Recommendation: [Suggestions for recommendation to improve the solution if its Overall Rating is below 10]
        #     """,
        # },
        {"role": "user", "content": f"Generate a report for Problem: {problem_input}, Solution: {solution_input}"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Prompt tokenized")
    model_output = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print("Output created")
    st.text_area("Idea Evaluation Report:", value=model_output, key="model_output", height=100)

# def display_messages(user_input):
#     st.text_area("User", value=user_input, key="user_messages", height=100)

#     # Replace the following line with your own logic to generate bot responses
#     bot_response = generate_bot_response(user_input)

#     st.text_area("Bot", value=bot_response, key="bot_messages", height=100)

# def generate_bot_response(user_input):
#     # Replace this function with your own logic to generate bot responses
#     # For simplicity, this example just echoes the user's input
#     return f"Bot: {user_input}"

if __name__ == "__main__":
    main()
