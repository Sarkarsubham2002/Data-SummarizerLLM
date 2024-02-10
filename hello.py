
from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import json

import speech_recognition as sr

import whisper

import torchaudio
import os
import argparse
import time
from flask import Flask, request, jsonify
import json
import time
import io

import librosa







app = Flask(__name__)
@app.route('/upload', methods=['GET','POST'])
def upload_audio():
    
        transcribed_text= ""
        if request.method == 'POST':
            audio_file = request.files['audioFile']
        else:
            audio_file = ""


        if audio_file:

            
            audio_file = request.files['audioFile']

            # Read audio data from the uploaded file
            audio_data = audio_file.read()


            audio_ndarray, sample_rate = torchaudio.load((audio_data))
            # audio_ndarray = waveform.numpy()
        

            # Load the Whisper ASR model
            model = whisper.load_model("base")

            # Transcribe the audio data using Whisper ASR
            result = model.transcribe(audio_ndarray.numpy())
            # Extract the transcribed text
            transcribed_text = result['text']

        return render_template('audio.html',transcribed_text= transcribed_text)


if __name__ == '__main__':
    
    app.run(debug=True)
