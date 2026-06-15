@st.cache_resource
def load_model():
    # Using the cloud-friendly base model for deployment
    model_path = "microsoft/DialoGPT-small" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model
  
