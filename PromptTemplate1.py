import os
from constants import OPENAI_KEY
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

import streamlit as st

#Streamlit Framework
st.title('Langchain Demo with OpenAI API')
input_text = st.text_input("Search the topic")

#OPenAI LLM
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
llm = OpenAI(temperature=0.8)

#Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

chain = LLMChain(llm=llm, prompt = first_input_prompt, verbose = True)


if input_text:
    st.write(chain.run(input_text))


