from transformers import pipeline, AutoTokenizer
import pyttsx3
import speech_recognition as sr
model = 'sharveshsubhash/llama-2-7b-miniguanaco-sharvesh-subhash'
tokenizer = AutoTokenizer.from_pretrained(model)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

reco = sr.Recognizer()
user = []
with sr.Microphone() as source:
  print("Ask me anything")
  audio = reco.listen(source)
try:
  text = reco.recognize_google(audio)
  user.append(text)
finally:
  result = pipe(f"[INST] {user} [/INST]")
  LMoutput = result[0]['generated_text'].split("[/INST]")[1]

engine = pyttsx3.init()
engine.say(LMoutput)
engine.runAndWait()