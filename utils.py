import os
import boto3
import pandas as pd
import numpy as np
import json
from openai import OpenAI
import streamlit as st
from botocore.exceptions import ClientError

openai_key = st.secrets["OPENAI_API_KEY"]
ACCESS_KEY_ID=st.secrets["ACCESS_KEY_ID"]
SECRET_ACCESS_KEY=st.secrets["SECRET_ACCESS_KEY"]

openai = OpenAI(
    api_key=openai_key,
)


def use_gpt(input_prompt):
        
    model_to_use = "gpt-4o"

    response = openai.chat.completions.create(
      model=model_to_use,
    #   prompt=input_prompt,
      messages = [ # Change the prompt parameter to messages parameter
            {"role": "system", "content": "You are a helpful assistant and you only respond in python dictionary."},
            {"role": "user", "content": input_prompt},
        ],
      temperature=0,
    #   max_tokens=1000,
    #   top_p=1,
    #   frequency_penalty=0.0,
    #   presence_penalty=0.0
    )
    # response_text = response['choices'][0]['text']
    response_text = response.choices[0].message.content
    return response_text

NEW_ENDPOINT_NAME = st.secrets["NEW_ENDPOINT"]
runtime= boto3.client('runtime.sagemaker',
                      aws_access_key_id=ACCESS_KEY_ID,
                      aws_secret_access_key=SECRET_ACCESS_KEY,
                      region_name='us-east-2')

bedrock = boto3.client('bedrock-runtime', 
                       aws_access_key_id=ACCESS_KEY_ID,
                      aws_secret_access_key=SECRET_ACCESS_KEY,
                      region_name='us-east-2')



def new_model_predict(wallet_csv, endpoint= NEW_ENDPOINT_NAME):
    response = runtime.invoke_endpoint(EndpointName=endpoint,
                                       ContentType='text/csv',
                                       EnableExplanations='`true`',
                                       Body=wallet_csv)
                                       
    # print(response)
    return response

def new_parse_response(query_response):
    response = query_response["Body"].read()
    
    response = response.decode("utf-8")
    response = json.loads(response)
    predictions = response["predictions"]["data"]
    explanations = response["explanations"]["kernel_shap"]
    
    # print(predictions)
    # print(explanations)
    record_output = explanations[0]
    record_shap_values = []
    if record_output is not None:
        for feature_attribution in record_output:
            record_shap_values.append(
                feature_attribution["attributions"][0]["attribution"][0]
            )
    
    # print(record_shap_values)
    # predicted_probabilities =  [predictions]
    return np.array([float(predictions)]), record_shap_values

def use_bedrock(input_prompt):
    model_id = "meta.llama3-3-70b-instruct-v1:0"
    # Format the request payload using the model's native structure.
    native_request = {
        "temperature": 0.5,
        # "messages": [
        #      {"role": "system", "content": "You are a helpful assistant and you only respond in python dictionary."},
        #     {
        #         "role": "user",
        #         "content": [{"type": "text", "text": input_prompt}],
        #     }
        # ],

        "prompt": f"{input_prompt}. Only respond with a python dictionary"
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = bedrock.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    # st.write(f"Response: {model_response['generation'][10:]}")
    response_text = model_response["generation"]
    return response_text