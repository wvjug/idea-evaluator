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
    st.text_area("Evaluation metrics:", value=clean_labels, key="bert_output", height=110)
    st.text_area("Example solution :", value=text_response, key="gpt_output", height=110)

    if problem_input != "" and solution_input != "" :
    # Run model and display output
        model_output = eval(problem_input, solution_input, model_ckpt="bert_classifier.pth")
        labels = model_output[0]
        average_score = sum(labels) / 3
        average_score = round(average_score, 2)
        clean_labels = f"Adherence to Circular Economy: {labels[0]} \n Market Potential: {labels[1]} \n Feasibility: {labels[2]} \n Average Score: {average_score}"
        st.text_area("Evaluation metrics:", value=clean_labels, key="bert_output", height=110)

        text_response = generate_response(problem_input)
        st.text_area("Example solution :", value=text_response, key="gpt_output", height=110)

if __name__ == "__main__":
    main()
