import pandas as pd 
import torch 
from sentence_transformers import SentenceTransformer, util 
import numpy as np 
import streamlit as st 
 
data = pd.read_csv('clothing_order_faq.csv') 
df = pd.DataFrame(data) 
print('Total FAQs loaded: ',len(df)) 
print(df.head()) 
 
model = SentenceTransformer('all-MiniLM-L6-v2') 
 
faq_embeddings = model.encode( 
    df['question'].tolist(), 
    convert_to_tensor = True, 
    show_progress_bar = True 
) 
print(f"Embedding shape: {faq_embeddings.shape}") # 20, 384 
 
def get_answer(user_query, threshold = 0.4): 
 
    query_embeddings = model.encode(user_query, convert_to_tensor = True) 
 
    similarities = util.cos_sim(query_embeddings, faq_embeddings)[0] 
 
    best_idx = torch.argmax(similarities).item() 
    best_score = similarities[best_idx].item() 
 
    if best_score > threshold: 
 
        matched_question = df.iloc[best_idx]['question'] 
        answer = df.iloc[best_idx]['answer'] 
 
        return { 
            'answer' : answer, 
            'matched_question' : matched_question, 
            'confidence' : best_score, 
            'found' : True 
        } 
 
    else: 
        return { 
            'answer' : "Sorry, I didn't understand. Please contact support@example.com", 
            'matched_question' : None, 
            'confidence' : best_score, 
            'found' : False 
        } 


st.title("FAQ Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
user_input = st.chat_input("Ask your question...")

if user_input:

    st.session_state.messages.append({
        "role" : "user",
        "content" : "user_input"
    })

    with st.chat_message("user"):
        st.write(user_input)
        
    result = get_answer(user_input)

    if result['found']:
        answer = result['answer']

    else:
        answer = result['answer']

    with st.chat_message("assistant"):
         st.write(answer)
 
         if not result['found']:
             st.write(f"(Confidence was only {result['confidence']:.3f})")
 
    st.session_state.messages.append({
         "role": "assistant",
         "content": answer
         })
