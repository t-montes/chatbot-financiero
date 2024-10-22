# ************************* Chatbot Financiero GenAI *************************
import google.cloud.bigquery as BigQuery
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import BigQueryLoader
from langchain.schema import format_document
from unidecode import unidecode
from datetime import datetime, date
import streamlit as st
import random
import utils
import json
import re

# ************************ Global and Config Variables ************************
with open("config/global.json", "r", encoding="utf-8") as variables:
    VARIABLES = json.load(variables)

with open("variables/examples.txt", "r", encoding="utf-8") as examples:
    EXAMPLES = examples.read()

with open("variables/assistant_response.json", "r", encoding="utf-8") as ass_response:
    ASSISTANT_RESPONSE = json.load(ass_response)

with open("variables/sql_prompt.txt", "r", encoding="utf-8") as prompt_template:
    PROMPT_TEMPLATE = prompt_template.read()

with open("variables/userq_type_prompt.txt", "r", encoding="utf-8") as userq_type_prompt:
    USERQ_TYPE_PROMPT = userq_type_prompt.read()

with open("variables/forecast_prompt.txt", "r", encoding="utf-8") as forecast_prompt:
    FORECAST_PROMPT = forecast_prompt.read()

with open("variables/talk_prompt.txt", "r", encoding="utf-8") as talk_prompt:
    TALK_PROMPT = talk_prompt.read()

PROJECT_ID = VARIABLES["global"]["project_id"]
DATASET_ID = VARIABLES["global"]["dataset_id"]
LOCATION_ID = VARIABLES["global"]["location_id"]
VERTEX_AI_MODEL = VARIABLES["global"]["vertex_ai_model"]
PCB_RESPONSES = ASSISTANT_RESPONSE["pcb_responses"]
PCB_NO_RESPONSES = ASSISTANT_RESPONSE["pcb_no_responses"]

CREDENTIALS = service_account.Credentials.from_service_account_file(VARIABLES["global"]["service_account_key"])
CREDENTIALS = CREDENTIALS.with_scopes([scope.strip() for scope in VARIABLES["global"]["authentication_scope"].split(',')])

BIGQUERY_CLIENT = BigQuery.Client(credentials=CREDENTIALS)
LLM = VertexAI(project=PROJECT_ID, location=LOCATION_ID, credentials=CREDENTIALS, model_name=VERTEX_AI_MODEL, max_output_tokens=8192, temperature=0)

# ************************ Functions ************************
def run_sql(clean_query):
    try:
        df = BIGQUERY_CLIENT.query(clean_query).result().to_dataframe()
        sql_error = ""
    except Exception as e:
        df = []
        sql_error = e
    return (sql_error, df)

def generate_and_display_sql(user_question, examples, context):
    SCHEMAS_QUERY = f"""
    SELECT table_catalog, table_schema, table_name, column_name, data_type
    FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`;
    """
    BQLOADER = BigQueryLoader(SCHEMAS_QUERY, page_content_columns=["table_catalog", "table_schema", "table_name", "column_name", "data_type"], credentials=CREDENTIALS)
    SCHEMAS_DOCS = BQLOADER.load()
    gemini_prompt = """
    Prompt:
    """ + PROMPT_TEMPLATE + """
    
    Esquema de tablas en BigQuery:
    {schemas_data}
    
    Ejemplos de Preguntas/SQL Bien Generados:
    """ + examples + """
    
    Contexto/Historial:
    """ + context + """
    
    Pregunta:
    """ + user_question
    
    chain = (
        {
            "schemas_data": lambda docs: "\n\n".join(
                format_document(doc, PromptTemplate.from_template("{page_content}"))
                for doc in docs
            ),
        } | PromptTemplate.from_template(gemini_prompt) | LLM
    )

    print("SCHEMAS")
    print("\n\n".join(
                format_document(doc, PromptTemplate.from_template("{page_content}"))
                for doc in SCHEMAS_DOCS
            ))
    
    # Process and Display Output
    result = chain.invoke(SCHEMAS_DOCS)
    clean_query = result.replace("```sql", "").replace("```", "")
    clean_query = unidecode(clean_query)
    return clean_query

def generate_nl_response_from_df(user_question, result_df, query=None):
    today = datetime.now()
    nl_prompt = f"""\
        Pregunta del usuario: {user_question}
        
        DataFrame: {result_df.to_string()}
        
        {'Query SQL que generó el DataFrame: ' + query if query else ''}
        
        Hoy es {today.strftime("%Y-%m-%d")}
        Genera una respuesta en lenguaje natural que resuma el resultado del DataFrame en función de la pregunta del usuario. NO menciones la consulta SQL, responde como si fueras un asistente; los precios dalos en $
    """
    natural_language_response = LLM(nl_prompt)
    natural_language_response = re.sub(r'[$€£¥₹]', r'\\$', natural_language_response)
    return natural_language_response

def generate_nl_response_from_ctx(user_question, context):
    if len(context) > 0:
        context = "Historial/Contexto:\n\n" + context
    else:
        context = ""
    nl_prompt = f"""\
        {TALK_PROMPT}
        
        {context}
        
        Pregunta del usuario: {user_question}
    """
    natural_language_response = LLM(nl_prompt)
    natural_language_response = re.sub(r'[$€£¥₹]', r'\\$', natural_language_response)
    return natural_language_response

def generate_userq_type(user_question):
    result = LLM(USERQ_TYPE_PROMPT + user_question)
    type_json = result.replace("```json", "").split("```")[0]
    type_json = json.loads(type_json)
    return type_json['query_type']

def generate_forecast_call(user_question):
    today = datetime.now()
    forecast_prompt = FORECAST_PROMPT.replace("{date}", today.strftime('%Y-%m-%d')) + user_question
    print(forecast_prompt)
    result = LLM(forecast_prompt)
    print(result)
    params_json = result.replace("```json", "").split("```")[0]
    params_json = json.loads(params_json)
    if 'error' in params_json: return params_json['error']

    #params_json['model'] = f"./ml/{params_json['model']}.pkl"
    up_date = datetime.strptime(params_json['up_date'], '%Y-%m-%d').date()
    if up_date > date(2026, 10, 31): return "Únicamente se pueden hacer predicciones hasta la fecha 2026-10-17" # 2 years after the end of the dataset
    diff = int(((up_date.year - today.year) * 12 + up_date.month - today.month + (up_date.day - today.day) / 30)//1)#+1
    if diff < 1: return "Para poder hacer predicciones, se deben poner fechas futuras a la fecha de hoy"
    params_json['steps'] = diff
    return params_json

def get_context_messages(n=3):
    context = []
    i,skipthis=0,False
    for message in st.session_state.messages[-2::-1]:
        if i//2 >= n: break
        if skipthis:
            skipthis=False
            continue
        if message["status"].startswith("success"): 
            i+=1
            context.append(f"{'Pregunta: ' if message['role'] == 'human' else 'Respuesta: '} {message['content']}")
        else: skipthis=True
    return '\n'.join(context[::-1])

# ************************ Main App ************************
# Build Frontend
st.set_page_config(layout="wide", page_title="ProCi | Chatbot-Financiero Demo", page_icon="./images/bot-logo.png")
with open("css/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
st.image('./images/header.png')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=('./images/user-logo.png' if message["role"] == 'human' else './images/bot-logo.png')):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "dataframe" in message:
                with st.expander("Ver la respuesta de la consulta SQL relacionada"):
                    st.dataframe(message["dataframe"], use_container_width=True, hide_index=True)
            if "sql_query" in message:
                with st.expander("Ver la consulta SQL relacionada"):
                    st.code(message["sql_query"], language="sql", line_numbers=True)

if prompt := st.chat_input("¡Déjame mostrar mi magia! ¿Cuál es tu pregunta?"):
    st.chat_message("human", avatar='./images/user-logo.png').markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt, "status": "success-human"})

    userq_type = generate_userq_type(prompt)
    with st.chat_message("assistant", avatar='./images/bot-logo.png'):
        if userq_type == 'prediction':
            with st.spinner("Llamando modelo de predicción..."):
                call_params = generate_forecast_call(prompt)
                if type(call_params) == str: 
                    st.write(call_params)
                    st.session_state.messages.append({"role": "assistant", "content": call_params, "status": "error-forecast"})
                else:
                    forecast = utils.predict(call_params['model'], call_params['steps'])
                    natural_language_response = generate_nl_response_from_df(prompt, forecast)
                    st.write(natural_language_response)
                    with st.expander("Ver la predicción"):
                        st.dataframe(forecast, use_container_width=True, hide_index=True)
                    st.session_state.messages.append({"role": "assistant", "content": natural_language_response, "dataframe": forecast, "status": "success-forecast"})
        elif userq_type == 'talk':
            with st.spinner("Procesando..."):
                context = get_context_messages(3)
                natural_language_response = generate_nl_response_from_ctx(prompt, context)
                st.write(natural_language_response)
                st.session_state.messages.append({"role": "assistant", "content": natural_language_response, "status": "success-talk"})
        else:
            with st.spinner("Generando consulta SQL..."):
                context = get_context_messages(3)
                clean_query = generate_and_display_sql(prompt, EXAMPLES, context)
                sql_error_result, result_df = run_sql(clean_query)
                if not sql_error_result or result_df:
                    natural_language_response = generate_nl_response_from_df(prompt, result_df, clean_query)
                    st.write(natural_language_response)
                    with st.expander("Ver la respuesta de la consulta SQL relacionada"):
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                    with st.expander("Ver la consulta SQL relacionada"):
                        st.code(clean_query, language="sql", line_numbers=True)
                    st.session_state.messages.append({"role": "assistant", "content": natural_language_response, "dataframe": result_df, "sql_query": clean_query, "status": "success-sql"})
                else:
                    print(f"SQL Error: {sql_error_result}")
                    cora_generated_no_response = random.choice(PCB_NO_RESPONSES)
                    st.write(cora_generated_no_response)
                    with st.expander("Ver la consulta SQL relacionada"):
                        st.code(clean_query, language="sql", line_numbers=True)
                    st.session_state.messages.append({"role": "assistant", "content": cora_generated_no_response, "sql_query": clean_query, "status": "error-sql"})
