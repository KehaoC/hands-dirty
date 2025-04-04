# chapter 1 pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a Hugging Face course my whole life.")
print(result)
