# integrate with openai
import os
from constant import openai_key
from langchain.llms import OpenAI
import streamlit as st
os.environ["OPENAI_API_KEY"]=openai_key

st.title('First project with langchain with openai')
input_text= st.text_input("search the topic you want")

# OPENAI LLMS
llm=OpenAI(temparature=0.7)

if input_text:
    st.write(llm(input_text))
