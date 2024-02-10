#!/usr/bin/env python3
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
    

    
from constants import CHROMA_SETTINGS


from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
from flask import Flask, request, jsonify
import json
import time


from flask import Flask, request, render_template

app = Flask(__name__)

CORS(app)


persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')

# this the model path of the lama model

model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
embeddings_model_name = "all-MiniLM-L6-v2"

model_n_ctx = 500
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))



# this one is the model path of the falcon model
model_path1 = os.environ.get('MODEL_PATH1')

result_generated = False


callbacks = [StreamingStdOutCallbackHandler()]
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/summarize', methods=['POST'])
def main():

    user_input = request.json

    if user_input is None:
        return


    text = ""
    result = ""

    text = user_input.get('text')

    result= summarizer(text, max_length=300, min_length=200, do_sample=False)
       

    response_data = {
        "result": result[0]['summary_text']
    }


    return jsonify(response_data)


@app.route('/', methods=['GET'])
def fun():

    response_data = {
        "result": "hello"
    }


    return jsonify(response_data)




@app.route('/grammar', methods=['POST'])
def new_page():
    
    user_input = request.json

    if user_input is None:
        return

    text = ""
    result = ""

    text = user_input.get('text')

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever) 
    result = ""     
    string1 ='correct the following sentence grammatically and with punctuation marks in relevent places : '
    text = string1+text      
    res = qa(text)
    answer= res['result']
    result = answer
    response_data = {
    "result": result
    }

    return jsonify(response_data)






if __name__ == '__main__':
    
    app.run(debug=True, host="0.0.0.0", port=5001)
