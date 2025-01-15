import ast
import streamlit as st
import pandas as pd
from utils import use_gpt, new_model_predict, new_parse_response

st.title("🎈 Transaction Monitoring Demo")


prompt = st.chat_input("Prompt for the kind of Dataset you want to generate")

input_sample = pd.read_csv("input_sample.csv", index_col=0)

if prompt:

    with st.spinner("Generating Dataset and Making Predictions ....."):
        input_prompt = f"""{prompt}. You can only generate one row of data, Categorical columns like channel, accountType can only be 0 or 1

        strictly output a python dictionary

        Example:
        {input_sample.to_dict('records')}
        """

        result = use_gpt(input_prompt)
        result = ast.literal_eval(result[10:-4])
        result_frame = pd.DataFrame([result])

        st.info("Generated Data")
        st.dataframe(result_frame.T,
                    use_container_width=True)
        

        NEW_ENDPOINT_NAME = "NewSigmaFraudProject-prod"
        endpoint = NEW_ENDPOINT_NAME

        query_response_batch = new_model_predict(
            result_frame.to_csv(header=False, index=False).encode("utf-8"), endpoint=endpoint
        )
        columns = result_frame.columns.tolist()
        predict_prob_batch, explanations = new_parse_response(query_response_batch)  # prediction probability per batch


        st.info(
            "Model Prediction")
        
        st.progress(predict_prob_batch[0], text=f"Fraud Probability {round(predict_prob_batch[0], 2)}")


