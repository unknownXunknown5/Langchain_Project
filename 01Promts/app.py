from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
load_dotenv()

# Note: The model name is spelled 'gemini-2.5-flash'
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.title("Study Notes Generator")
topic=st.text_input("Enter Your Topic")
file=st.file_uploader("Upload Your File") 
Difficulty=st.slider("Difficulty Level",0,5,2)
language=st.selectbox("Select Language",["Hindi","English","Hinglish"])
df=pd.read_csv("domain.txt")

domain=st.selectbox("Select Domain",df.iloc[:,0].tolist())

if topic:
    entopic=topic
else:
    entopic=file  

prompt=PromptTemplate(
    template="""
You are an expert {domain}.

Create study notes on the topic: {entopic}

Difficulty Level: {difficulty}
Language: {language}

Provide:

1. 150-word Summary
2. 10 Key Points
3. 5 Interview Questions with Answers
4. 5 MCQs with Correct Answers
5. Important Formulas (if applicable)
6. 5 Flashcards

Format everything neatly with headings.
"""
)

chain=prompt | model  

# REMOVED: result=chain.invoke({}) <- This line was causing the crash

if st.button("Generate"):
    # Ensure the user has actually provided a topic or file before running
    if entopic:
        result = chain.invoke({
            "difficulty": Difficulty,
            "domain": domain,
            "entopic": entopic,
            "language": language
        })
        st.write(result.content)
    else:
        st.warning("Please enter a topic or upload a file first.")

