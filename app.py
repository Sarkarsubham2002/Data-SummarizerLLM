#!/usr/bin/env python3
from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import json

import speech_recognition as sr

import whisper

import librosa


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
import io

from flask import Flask, request, render_template



from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import json

import speech_recognition as sr

import whisper
import os
import argparse
import time
from flask import Flask, request, jsonify
import json
import time
import io



import os
import speech_recognition as sr
import whisper
from transformers import pipeline



audio_file_path = "sage.wav"




model = whisper.load_model("tiny.en")


result = model.transcribe(audio_file_path)


transcribed_text = result['text']






Summ = summarizer(transcribed_text, max_length=400, min_length=80, do_sample=False)


print(Summ)



# app = Flask(__name__)




# # if not load_dotenv():
# #     print("Could not load .env file or it is empty. Please check if it exists and is readable.")
# #     exit(1)





# embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
# persist_directory = os.environ.get('PERSIST_DIRECTORY')

# model_type = os.environ.get('MODEL_TYPE')

# # this the model path of the lama model

# model_path = os.environ.get('MODEL_PATH')


# model_n_ctx = os.environ.get('MODEL_N_CTX')
# model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
# target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
# # this one is the model path of the falcon model
# model_path1 = os.environ.get('MODEL_PATH1')

# from constants import CHROMA_SETTINGS
# result_generated = False
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# @app.route('/', methods=['GET', 'POST'])
# def main():


#     if request.method == 'POST':
#         user_input = request.form['user_input']
#     else:
#         user_input = ""




#     result = ""

#     if request.method == 'POST':

#         query = request.form['user_input']


#         result= summarizer(query, max_length=400, min_length=80, do_sample=False)
       
        

        

#         response_data = {
#             "query": query,
#             "result": result
#         }

#         with open('data.json', 'w') as json_file:
#             json.dump(response_data, json_file)


#     return render_template('home.html',user_input=user_input, result=result)









# audio_directory = "audio_files"
# recognizer = sr.Recognizer()



# @app.route('/upload', methods=['GET','POST'])
# def upload_audio():
        
#         if request.method == 'POST':
#             audio_file = request.files['audioFile']
#         else:
#             audio_file = ""

#         transcribed_text= ""
        
#         Summ=""


#         if audio_file:
#             # Generate a unique filename
#             filename = os.path.join(audio_directory, audio_file.filename)
#             audio_file.save(filename)
#             audio_file_name = filename
#             with sr.AudioFile(audio_file_name) as source:
#                 audio_data = recognizer.record(source)
            
            
#                 model = whisper.load_model("base")
#                 result = model.transcribe(audio_data)

#                 # Extract the transcribed text
#                 transcribed_text = result['text']
            
#             Summ= summarizer(transcribed_text, max_length=400, min_length=80, do_sample=False)
        
            

                

#             response_data = {
#                 "query": transcribed_text,
#                 "result": Summ
#             }

#             with open('audio.json', 'w') as json_file:
#                 json.dump(response_data, json_file)


                            

            
#         return render_template('audio.html',transcribed_text= transcribed_text,audio=Summ)
   
    












# @app.route('/sent', methods=['GET', 'POST'])
# def new_page():
    


#     if request.method == 'POST':
#         user_input = request.form['user_input']
#     else:
#         user_input = ""
    
   
   
   
#     # Parse the command line arguments
#     args = parse_arguments()
#     embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
#     chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
#     db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
#     retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
#     # activate/deactivate the streaming StdOut callback for LLMs
#     callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
#     # Prepare the LLM
#     match model_type:
#         case "LlamaCpp":
#             llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#         case "GPT4All":
#             llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#         case _default:
#             # raise exception if model_type is not supported
#             raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    
    
    
    
    
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    
#     result = ""
    






#     if request.method == 'POST':
#         query = request.form['user_input']
        
#         # Your existing code to get the answer

#         # query= 'hello'
#         string1 ='correct the following sentence grammatically and with punctuation marks in relivent places : '
#         query = string1+query



#         # start = time.time()
#         res = qa(query)
#         answer, docs = res['result'], [] if args.hide_source else res['source_documents']
#         end = time.time()

#         result = answer

        

#         # Save the data to 'data.json'
#         response_data = {
#             "query": query,
#             "result": result
#         }

#         with open('sent.json', 'w') as json_file:
#             json.dump(response_data, json_file)

#     return render_template('sentence.html', user_input=user_input, result=result)


















# def parse_arguments():
#     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
#                                                  'using the power of LLMs.')
#     parser.add_argument("--hide-source", "-S", action='store_true',
#                         help='Use this flag to disable printing of source documents used for answers.')

#     parser.add_argument("--mute-stream", "-M",
#                         action='store_true',
#                         help='Use this flag to disable the streaming StdOut callback for LLMs.')

#     return parser.parse_args()






# if __name__ == '__main__':
    
#     app.run(debug=True)
