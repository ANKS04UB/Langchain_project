# integrate with openai
import os
from constant import openai_key
from langchain.llms import OpenAI
import streamlit as st
os.environ["OPENAI_API_KEY"]=openai_key
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

st.title('celebrity search result')
input_text= st.text_input("search the topic you want")

# propmt templates
prompt_input = PromptTemplate(input_variables=['name'],
                              template="tell me about celebrity {name}")

# Memory
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
discr_memory=ConversationBufferMemory(input_key='dob',memory_key='discription_history')

# OPENAI LLMS
llm=OpenAI(temparature=0.7)
chain=LLMChain(llm=llm,prompt=prompt_input,verbose=True,output_key='person',memory=person_memory)

second_prompt_input = PromptTemplate(input_variables=['person'],
                              template="when was  {person} born")
chain2=LLMChain(llm=llm,prompt=second_prompt_input,verbose=True,output_key='dob',memory=dob_memory)


third_prompt_input = PromptTemplate(input_variables=['person'],
                              template="mention 5 major events happened around  {dob} in the world")
chain3=LLMChain(llm=llm,prompt=third_prompt_input,verbose=True,output_key='description',memory=discr_memory)

parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander("Person name"):
        st.info(person_memory.buffer)

    with st.expander("Major events"):
        st.info(discr_memory.buffer)
