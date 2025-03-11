import os
import random
import numpy as np
import pickle
import json

# Ensure comprehensive NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full file paths
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
WORDS_PATH = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.keras")

def safe_tokenize(text):
    """
    Safely tokenize text with fallback method
    """
    try:
        # Try word_tokenize first
        tokens = word_tokenize(str(text), language='english')
    except Exception:
        # Fallback to simple split if tokenize fails
        tokens = str(text).split()
    return [token.lower() for token in tokens]

def prepare_training_data():
    # Initialize lists
    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!", ".", ","]

    # Read intents file
    try:
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            intents = json.load(f)
    except Exception as e:
        print(f"Error reading intents file: {e}")
        return [], [], [], []

    # Process intents
    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            # Tokenize words safely
            w = safe_tokenize(pattern)
            
            # Skip empty patterns
            if not w:
                continue

            words.extend(w)
            
            # Add to documents
            documents.append((w, intent["tag"]))

            # Add to classes
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # Lemmatize and clean words
    words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # Create word to index mapping
    word_to_index = {word: i for i, word in enumerate(words)}

    # Prepare training data with consistent dimensionality
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        # Create bag of words with consistent length
        bag = [0] * len(words)
        for word in doc[0]:
            if word in word_to_index:
                bag[word_to_index[word]] = 1

        # Create output row
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append((bag, output_row))

    # Separate features and labels
    train_x = [item[0] for item in training]
    train_y = [item[1] for item in training]

    # Convert to numpy arrays
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Save preprocessed data
    try:
        pickle.dump(words, open(WORDS_PATH, "wb"))
        pickle.dump(classes, open(CLASSES_PATH, "wb"))
    except Exception as e:
        print(f"Error saving pickle files: {e}")

    return train_x, train_y, words, classes

def train_model(train_x, train_y):
    # Ensure input shape is consistent
    input_shape = train_x.shape[1]

    # Create model architecture
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(train_y.shape[1], activation="softmax")
    ])

    # Compile model
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # Train model
    hist = model.fit(
        train_x, 
        train_y, 
        epochs=200, 
        batch_size=5, 
        verbose=1
    )

    # Save model explicitly in .keras format
    try:
        print(f"Saving model to: {MODEL_PATH}")
        model.save(MODEL_PATH, save_format='keras')
        print(f"Model saved successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def main():
    # Prepare training data
    train_x, train_y, words, classes = prepare_training_data()


    
    # Check if we have valid training data
    if train_x.size == 0 or train_y.size == 0:
        print("No training data found. Please check your intents.json file.")
        return
    
    # Print some information
    print(f"Total documents: {len(train_x)}")
    print(f"Number of classes: {len(classes)}")
    print(f"Number of unique words: {len(words)}")
    print(f"Training data shape: {train_x.shape}")

    # Train the model
    train_model(train_x, train_y)

if __name__ == "__main__":
    main()