import os
import random
import numpy as np
import pickle
import json
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import nltk
nltk.download('punkt')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full file paths
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.keras")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
WORDS_PATH = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.pkl")

lemmatizer = WordNetLemmatizer()

def load_chatbot_data():
    """Load all required chatbot data files"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            intents = json.load(f)
        print("Intents loaded successfully")
        
        with open(WORDS_PATH, 'rb') as f:
            words = pickle.load(f)
        print("Words loaded successfully")
        
        with open(CLASSES_PATH, 'rb') as f:
            classes = pickle.load(f)
        print("Classes loaded successfully")
        
        return model, intents, words, classes
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available files: {os.listdir(BASE_DIR)}")
        raise

# Load the data
model, intents, words, classes = load_chatbot_data()

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    tag = ints[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm not sure I understand!"

def check_files_exist():
    """Check if all required files exist"""
    required_files = {
        "Model": MODEL_PATH,
        "Intents": INTENTS_PATH,
        "Words": WORDS_PATH,
        "Classes": CLASSES_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name} file not found at: {path}")
    
    if missing_files:
        print("\nMissing required files:")
        for msg in missing_files:
            print(f"- {msg}")
        print("\nPlease run train.py first to generate the required files.")
        raise FileNotFoundError("Required files are missing")

# Chat function
def chat():
    print("Chatbot is running! (type 'quit' to stop)")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Goodbye!")
            break
        ints = predict_class(message, model)
        response = get_response(ints, intents)
        print(f"Bot: {response}")

def main():
    try:
        print("Checking required files...")
        check_files_exist()
        print("Starting chatbot initialization...")
        chat()
    except Exception as e:
        print(f"\nError during chatbot execution: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available files: {os.listdir(BASE_DIR)}")
        raise

if __name__ == "__main__":
    main()
