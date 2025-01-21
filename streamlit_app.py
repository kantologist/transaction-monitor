import ast
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils import use_gpt, new_model_predict, new_parse_response, use_bedrock


explanation_dic = {
    'receiver_prev_count': 'Number of previous transactions involving the receiver.',
    'channel_api': 'Transaction occurred through API channel (1 if True, 0 otherwise).',
    'channel_bank transfer': 'Transaction occurred through bank transfer (1 if True, 0 otherwise).',
    'channel_null': 'Transaction channel is unknown or null (1 if True, 0 otherwise).',
    'channel_pos': 'Transaction occurred through Point of Sale (POS) (1 if True, 0 otherwise).',
    'channel_vas': 'Transaction occurred through Value Added Service (VAS) (1 if True, 0 otherwise).',
    'amount': 'The amount of the transaction in naira.',
    'trans_deviation': 'Deviation of the transaction amount in naira from the user\'s typical behavior.',
    'isExternalPayment': 'Indicates if the transaction is an external payment (1 if True, 0 otherwise).',
    'status': 'Status of the transaction (e.g., success, failed).',
    'type_credit': 'Transaction is a credit (1 if True, 0 otherwise).',
    'type_debit': 'Transaction is a debit (1 if True, 0 otherwise).',
    'secs_after_midnight': 'Seconds elapsed since midnight at the time of transaction.',
    'preBalance': 'Account balance in naira before the transaction.',
    'aggregate_30_amount_sum': 'Sum of transaction amounts in naira in the last 30 days.',
    'aggregate_30_preBalance_sum': 'Sum of account balances in naira before transactions in the last 30 days.',
    'aggregate_30_amount_mean': 'Average transaction amount in naira in the last 30 days.',
    'aggregate_30_preBalance_mean': 'Average account balance before transactions in the last 30 days.',
    'aggregate_30_count': 'Number of transactions in the last 30 days.',
    'aggregate_60_amount_sum': 'Sum of transaction amounts in naira in the last 60 days.',
    'aggregate_60_preBalance_sum': 'Sum of account balances in naira before transactions in the last 60 days.',
    'aggregate_60_amount_mean': 'Average transaction amount in nai in the last 60 days.',
    'aggregate_60_preBalance_mean': 'Average account balance before transactions in the last 60 days.',
    'aggregate_60_count': 'Number of transactions in the last 60 days.',
    'recent_transactions': 'Number of recent transactions within a short time window.',
    'firstTransfer': 'Indicates if this is the user\'s first transaction (1 if True, 0 otherwise).',
    'transactionLocationLat': 'Latitude of the transaction location.',
    'transactionLocationLong': 'Longitude of the transaction location.',
    'accountType_business': 'Account is a business account (1 if True, 0 otherwise).',
    'accountType_business-customer': 'Account is a business-customer account (1 if True, 0 otherwise).',
    'accountType_customer': 'Account is a customer account (1 if True, 0 otherwise).',
    'accountType_corporate': 'Account is a corporate account (1 if True, 0 otherwise).',
    'accountType_individual': 'Account is an individual account (1 if True, 0 otherwise).',
    'isPhoneNumberVerified': 'Indicates if the user\'s phone number is verified (1 if True, 0 otherwise).',
    'isBanned': 'Indicates if the user is banned (1 if True, 0 otherwise).',
    'age': 'Age of the user.',
    'secs_after_user_creation': 'Seconds elapsed since the user account was created.',
    'isIdentityVerified': 'Indicates if the user\'s identity is verified (1 if True, 0 otherwise).'
}

EXEMPTED_FEATURES = ['transactionLocationLat', 'transactionLocationLong']

def predict(edited_frame):
    # st.write(edited_frame["receiver_prev_count"])
    NEW_ENDPOINT_NAME = "NewSigmaFraudProject-prod"
    endpoint = NEW_ENDPOINT_NAME

    query_response_batch = new_model_predict(
        edited_frame.to_csv(header=False, index=False).encode("utf-8"), endpoint=endpoint
    )
    columns = edited_frame.columns.tolist()
    predict_prob_batch, explanations = new_parse_response(query_response_batch)  # prediction probability per batch

    explanation = ""
    if predict_prob_batch[0] > 0.3:
        
        edited_frame["secs_after_midnight"] = edited_frame["secs_after_midnight"] // 3600
        edited_frame["secs_after_user_creation"] = edited_frame["secs_after_user_creation"] // 3600
        
        explanation_dic["secs_after_midnight"] = 'Hours elapsed since midnight at the time of transaction.'
        explanation_dic["secs_after_user_creation"] =  'Hours elapsed since the user account was created.'
        
        
        
        explanation =  [(explanation_dic[i],j, edited_frame[i].item()) for i,j in zip(columns, explanations) if i not in EXEMPTED_FEATURES]
    
        sorted_explanation = sorted(explanation, key=lambda item: item[1], reverse=True)[:5]
        
        descriptions = [ f"{i[0]}: {str(i[2])}" for i in sorted_explanation]
        
        # print(descriptions)

        input_prompt = f"""Given the SHAP outputs below, generate a natural language explanation for why a transaction was flagged and might need to be reviewed for fraud. The explanation should include the following disclaimer at the top:
        'This explanation is based on a machine learning model, which considers a combination of multiple factors. No single factor is definitive; the flagging results from the interplay of these factors.' 
        The explanation should be brief, succinct, and easy to read. Use the provided SHAP outputs, which include factors and their contributions (positive or negative), to create the description.
        SHAP Outputs:
        {descriptions} 
        
        Output Example:
        'This transaction was flagged because [factor_1] had a significant impact, which is [value_1], combined with [factor_2], which is [value_2]. Other factors, such as [factor_3], [factor_4] and [factor_5], also played a role in raising the alert.'
        Note that credit and debit means direction of payments not cards."""
        
        # input_prompt = f"all the factors {','.join(descriptions)} and their values combined to an explanation for why the transaction might need to be reviewed for fraud as an explanation, showing the values. Credit and debit means direction of payments not cards"
        response = use_gpt(input_prompt)
        explanation  = response

    st.info("Model Prediction")
    # st.write(type(explanation))
    # explanation = ast.literal_eval(explanation)

    if predict_prob_batch[0] < 0.3:
        st.success(f"Looks Safe ({round(predict_prob_batch[0], 3) * 100} %)")
        # st.progress(predict_prob_batch[0], text=f"{round(predict_prob_batch[0], 2)}")
    elif predict_prob_batch[0] >= 0.5:
        st.error(f"High Risk ({round(predict_prob_batch[0], 3) * 100} %) ")
        st.progress(predict_prob_batch[0], text=f"{round(predict_prob_batch[0], 2)}")
        # st.error(explanation[32:-7])
    else:
        st.warning(f"Medium Risk ({round(predict_prob_batch[0], 3) * 100} %)")
        # st.progress(predict_prob_batch[0], text=f"{round(predict_prob_batch[0], 2)}")
        # st.warning(explanation[32:-7])

st.title("Transaction Monitoring Simulator")

# col1, col2= st.columns(2)

BEDROCK = False

# with col1:
#     prompt = st.chat_input("Prompt for the kind of Dataset you want to generate")

# with col2: 
#     option = st.selectbox(
#         "Select Model",
#         ("CHAT GPT", "BEDROCK"),
#     )

with st.container(border=True):
    prompt = st.chat_input("Prompt for the kind of Dataset you want to generate")
    option = st.selectbox(
        "Model",
        ("Chat GPT", "Amazon BEDROCK"),
    )

if option == "Amazon BEDROCK":
    BEDROCK = True

input_sample = pd.read_csv("input_sample.csv", index_col=0)

if prompt:
    # if 'result_frame' in st.session_state:
    #     del st.session_state.result_frame

    @st.fragment
    def predict_spinner():
        with st.spinner("Generating Dataset and Making Predictions ....."):
            input_prompt = f"""{prompt}. You can only generate one row of data. 

            Using this explanation to understand the variables {explanation_dic}. 

            strictly output a python dictionary. 

            Example:
            {input_sample.to_dict('records')}
            """

            if BEDROCK:
                result = use_bedrock(input_prompt)
                result = ast.literal_eval(result[10:])
            else:
                result = use_gpt(input_prompt)
                result = ast.literal_eval(result[10:-4])

            # st.write(result)
            if type(result) == list:
                result = result[0]
            

            st.info("Prompt")
            st.code(f"{prompt}")
            st.info("Generated Data")

            # def form_callback():
            #     st.session_state['result_frame'] = edited_frame

            # with st.form("my_form"):
            # if 'result_frame' in st.session_state:
            #     st.write("found saved state")
            #     result_frame = st.session_state['result_frame']
            #     st.write(result_frame["receiver_prev_count"])
            # else:
            #     st.write("No saved state!") 
            result_frame = pd.DataFrame([result])

            # st.write(result_frame)

            transactionData = {
                "reference" : "2303pee2fc",
                "amount" : result_frame["amount"].item(),
                "receiverAccount" : "93829392233",
                "isExternalPayment" : result_frame["isExternalPayment"].item(),
                # "status" : True if result_frame["status"] else False,
                "status" : result_frame["status"].item(),
                "senderAccount" : "01929393923",
                "balanceBefore" : result_frame["preBalance"].item(),
                "type" : result_frame["type_debit"].item(),
                "channel" : "pos",
                "transactionDate" : datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                "vasReceiver": "08124668857",
                "currency" : "ngn",
                "narration" : "transaction generated by chat gpt",
                "isInternalAccount": False
            }

            # st.write(transactionData)

            # Transaction Type
            if result_frame["type_credit"].item():
                transactionData["type"] = "credit"
            if result_frame["type_debit"].item():
                transactionData["type"] = "debit"


            # Transaaction Channel
            if result_frame["channel_api"].item():
                transactionData["channel"] = "api"
            if result_frame["channel_bank transfer"].item():
                transactionData["channel"] = "bank_transfer"
            if result_frame["channel_null"].item():
                transactionData["channel"] = "null"
            if result_frame["channel_pos"].item():
                transactionData["channel"] = "pos"
            if result_frame["channel_vas"].item():
                transactionData["channel"] = "vas"

            
            anonymizedUserData = {
                "uniqueId" : "e8baeb9c-e563-11ed-b5ea-0242ac120002",
                "accountType" : "individual",
                "businessCategory" : "retail",
                "isPhoneNumberVerified" : result_frame["isPhoneNumberVerified"].item(),
                "isBanned" : result_frame["isBanned"].item(),
                "dateJoined" : "2022-01-01 23:58:00",
                "age" : result_frame["age"].item(),
                "isIdentityVerified" : result_frame["isIdentityVerified"].item(),
                "state" : "lagos",
                "city" : "ikeja",
                "country" : "Nigeria"
            }

            secs_ = result_frame["secs_after_user_creation"].item()
            anonymizedUserData["dateJoined"] = (datetime.today() - timedelta(0,int(secs_))).strftime('%Y-%m-%d %H:%M:%S')

            if result_frame["accountType_business"].item():
                anonymizedUserData["accountType"] = "business"
            if result_frame["accountType_business-customer"].item():
                anonymizedUserData["accountType"] = "business-customer"
            if result_frame["accountType_corporate"].item():
                anonymizedUserData["accountType"] = "corporate"
            if result_frame["accountType_customer"].item():
                anonymizedUserData["accountType"] = "customer"
            if result_frame["accountType_individual"].item():
                anonymizedUserData["accountType"] = "individual"

            location = {"latitude": result_frame["transactionLocationLat"].item(),
                        "longitude": result_frame["transactionLocationLong"].item()}

            
            # edited_frame = st.dataframe(result_frame.T,
            #         use_container_width=True)
            
            st.write("Transaction Data")
            st.dataframe(transactionData,
                    use_container_width=True)
            
            st.write("anonymized User Data")
            st.dataframe(anonymizedUserData,
                    use_container_width=True)
            
            st.write("location")
            st.dataframe(location, use_container_width=True)

            history = {
                'Sum of transaction amounts in naira in the last 30 days': result_frame['aggregate_30_amount_sum'].item(), 
                'Sum of account balances in naira before transactions in the last 30 days': result_frame['aggregate_30_preBalance_sum'].item(),
                'Average transaction amount in naira in the last 30 days.': result_frame['aggregate_30_amount_mean'].item(),
                'Average account balance before transactions in the last 30 days': result_frame['aggregate_30_preBalance_mean'].item(),
                'Number of transactions in the last 30 days.': result_frame['aggregate_30_count'].item(),
                'Sum of transaction amounts in naira in the last 60 days.': result_frame['aggregate_60_amount_sum'].item(),
                'Sum of account balances in naira before transactions in the last 60 days': result_frame['aggregate_60_preBalance_sum'].item(),
                'Average transaction amount in naira in the last 60 days': result_frame['aggregate_60_amount_mean'].item(),
                'Average account balance before transactions in the last 60 days': result_frame['aggregate_60_preBalance_mean'].item(),
                'Number of transactions in the last 60 days': result_frame['aggregate_60_count'].item()
            }

            # history = pd.DataFrame([history])
            st.write("History")
            st.dataframe(history, use_container_width=True)


            
            # edited_frame = edited_frame.T
                # st.session_state['result_frame'] = edited_frame
                # submitted = st.form_submit_button("Make Prediction", on_click=form_callback)

            # if submitted:
            #     st.session_state['result_frame'] = edited_frame

            
            predict(result_frame)

        # if st.button("Predict", type='primary'):
        #     predict(edited_frame)
        # else:
        #     predict(edited_frame)
    predict_spinner()

# if st.button("Test", type='primary'):
#     st.write("Clicked stuff")
# else:
#     st.write("not Clicked!")


