# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import AutoTokenizer
from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_name = 's-1-n-t-h/bart-cnn-optimised'
    model = ORTModelForSeq2SeqLM.from_pretrained(model_name,from_transformers=True,use_auth_token='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')
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token='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')
    #pipeline('summarization', model="facebook/bart-large-cnn", framework='pt',use_auth_token='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')

if __name__ == "__main__":
    download_model()
