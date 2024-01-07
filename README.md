# Idea Evaluator

Our solution is a web app that evaluates the quality of a solution given a problem in the context of the circular economy. The quality of the solution is evaluated using 3 metrics from a scale of 1-5 using a BERT model. It then recommends improvements to the solution based on this evaluation using a GPT model. The metrics chosen are the following: 
##
#### 1. Adherence to Circular Economy
###
Fit into the three principles:
1) Waste and Pollution Elimination
2) Product Circulation
3) Nature Regeneration
##
#### 2. Market Potential
###
- Market size and saturation
- Current demands
- Growth potential
##
#### 3. Feasibility
- Implementation Difficulties and Risks
##
## Dependencies
The app is built on the Python 3 library Streamlit and requires the following packages to be run:
1) Streamlit (v1.29.0)
2) PyTorch
3) Transformers by Hugging Face
##
## Running Instructions
1. Install dependencies
2. Run Streamlit app using
```
#bash
streamlit run chat_ui.py
```
3. This will run the app on localhost and open it on the default browser