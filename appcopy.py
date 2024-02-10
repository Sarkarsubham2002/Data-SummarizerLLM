#!/usr/bin/env python3
from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import json
    


from langchain.callbacks.manager import CallbackManager

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







# if not load_dotenv():
#     print("Could not load .env file or it is empty. Please check if it exists and is readable.")
#     exit(1)





embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"

model_type ="GPT4All"
# this the model path of the lama model

model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"


model_n_ctx = 500
model_n_batch = 2
target_source_chunks = 2



# this one is the model path of the falcon model
model_path1 = os.environ.get('MODEL_PATH1')

from constants import CHROMA_SETTINGS




result_generated = False


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/', methods=['GET', 'POST'])
def main():


    if request.method == 'POST':
        user_input = request.form['user_input']
    else:
        user_input = ""




    result = ""

    if request.method == 'POST':

        query = request.form['user_input']


        result= summarizer(query, max_length=400, min_length=80, do_sample=False)
       
        

        

        response_data = {
            "query": query,
            "result": result
        }

        with open('data.json', 'w') as json_file:
            json.dump(response_data, json_file)


    return render_template('home.html',user_input=user_input, result=result)












@app.route('/sent', methods=['GET', 'POST'])
def new_page():
    


    if request.method == 'POST':
        user_input = request.form['user_input']
    else:
        user_input = ""
    
   
   
   
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    callback_manager = CallbackManager(callbacks)
   
    n_gpu_layers = 15  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 70  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    llm = LlamaCpp(
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            model_path=model_path,
            max_tokens=1000,
            callback_manager=callback_manager,
            
            verbose=False, # Verbose is required to pass to the callback manager,
            temperature=0.8,
            n_ctx=500
    )
    
    
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    
    result = ""
    






    if request.method == 'POST':
        query = request.form['user_input']
        
        # Your existing code to get the answer

        # query= 'hello'
        string1 ='correct the following sentence grammatically and with punctuation marks in relivent places : '
        query = string1+query



        # start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        result = answer

        

        # Save the data to 'data.json'
        response_data = {
            "query": query,
            "result": result
        }

        with open('sent.json', 'w') as json_file:
            json.dump(response_data, json_file)

    return render_template('sentence.html', user_input=user_input, result=result)


















def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()






if __name__ == '__main__':
    
    app.run(debug=False)
