from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, tokenizer
    
    device = 0 if torch.cuda.is_available() else -1
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained('braindao/flan-t5-cnn',from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained('braindao/flan-t5-cnn')
    model = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result
