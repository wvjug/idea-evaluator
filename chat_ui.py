import streamlit as st
from bert_classifier.eval import eval

def main():
    st.title("Solution Evaluator")

    # Input box for user to enter messages
    problem_input = st.text_input("Enter the Problem you want to solve:", "")
    solution_input = st.text_input("Enter the Solution to evaluate:", "")

    if problem_input != "" and solution_input != "" :
    # Run model and display output
        model_output = eval(problem_input, solution_input, model_ckpt="bert_classifier.pth")
        labels = model_output[0]
        clean_labels = f"Circular Economy Impact: {labels[0]} \n Market Potential: {labels[1]} \n Feasibility: {labels[2]}"
        st.text_area("Idea Evaluation Report:", value=clean_labels, key="model_output", height=100)

if __name__ == "__main__":
    main()
