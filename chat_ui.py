import streamlit as st
from bert_classifier.eval import eval
from gpt_chatbot.tinyllama import generate_response

def main():
    st.title("Solution Evaluator")

    # Input box for user to enter messages
    problem_input = st.text_input("Enter the Problem you want to solve:", "")
    solution_input = st.text_input("Enter the Solution to evaluate:", "")

    # Initial UI components
    clean_labels, text_response = "", ""

    if problem_input != "" and solution_input != "" :
    # Run model and display output
        model_output = eval(problem_input, solution_input, model_ckpt="bert_classifier.pth")
        labels = model_output[0]
        average_score = sum(labels) / 3
        average_score = round(average_score, 2)

        st.write("Evaluation metrics:")
        st.write(f"Adherence to Circular Economy: {labels[0]}")
        st.write(f"Market Potential: {labels[1]}")
        st.write(f"Feasibility: {labels[2]}")
        st.write(f"Average Score: {average_score}")

        text_response = generate_response(problem_input)
        st.write("Example alternative solution:")
        st.write(text_response)

if __name__ == "__main__":
    main()
