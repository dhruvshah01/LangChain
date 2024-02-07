import os
from constants import OPENAI_KEY
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

import streamlit as st

st.title('Langchain Demo with OpenAI API')
input_text = st.text_input("Search the topic")

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))


