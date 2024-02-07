import os
from constants import OPENAI_KEY
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

#Streamlit Framework
st.title('Langchain Demo with OpenAI API')
input_text = st.text_input("Search the topic")

#OPenAI LLM
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
llm = OpenAI(temperature=0.8)

#Memory
person_mem = ConversationBufferMemory(input_key = 'name', memory_key = 'chat_history')
dob_mem = ConversationBufferMemory(input_key = 'person', memory_key = 'chat_history')
desc_mem = ConversationBufferMemory(input_key = 'dob', memory_key = 'chat_history')

#Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

chain = LLMChain(llm=llm, prompt = first_input_prompt, verbose = True, output_key = 'person', memory = person_mem)

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born"
)

chain2 = LLMChain(llm=llm, prompt = second_input_prompt, verbose = True, output_key = 'dob', memory = dob_mem)

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happened around {dob} in the world"
)


chain3 = LLMChain(llm=llm, prompt = third_input_prompt, verbose = True, output_key = 'desc', memory = desc_mem)

parentChain = SequentialChain(
    chains = [chain, chain2, chain3], 
    input_variables = ['name'], 
    output_variables=['person', 'dob'],
    verbose = True)

if input_text:
    st.write(parentChain({'name': input_text}))


