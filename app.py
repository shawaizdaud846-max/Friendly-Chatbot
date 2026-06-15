import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Configure the Streamlit page
st.set_page_config(page_title="Friendly Bot", page_icon="🤖", layout="centered")

st.title("Friendly Bot 🤖")
st.write("A casual, general-purpose chatbot running entirely locally via TensorFlow!")

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = "microsoft/DialoGPT-small" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# ---> THE FIX: We must call the function to actually load the model! <---
tokenizer, model = load_model()

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
if prompt := st.chat_input("Say something friendly..."):
    
    # 1. Display and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Bot Response
    with st.chat_message("assistant"):
        # Encode the prompt and append the end-of-sentence token
        input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='tf')
        
        # Generate raw token outputs from the model
        bot_output = model.generate(
            input_ids, 
            max_length=150, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,        # Allows for more natural, varied responses
            top_k=50,              
            top_p=0.95,            
            temperature=0.7        # Controls creativity (higher = more random)
        )
        
        # Decode the new tokens back into human-readable text
        response = tokenizer.decode(bot_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Fallback just in case the model generates empty text
        if not response.strip():
            response = "I'm not exactly sure what to say to that, but I'm happy we are chatting! 😊"
            
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
