
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



audio_file_path = "hello.wav"




model = whisper.load_model("tiny.en")


result = model.transcribe(audio_file_path)


transcribed_text = result['text']


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

Summ = summarizer(transcribed_text, max_length=2000, min_length=80, do_sample=False)


print(transcribed_text)
