# HotelBot - AI-Powered Hotel Booking Assistant 🏨🤖

An intelligent chatbot designed to streamline hotel room booking processes through Natural Language Processing (NLP) and deep learning techniques. Built with Python, TensorFlow/Keras, and NLTK, this chatbot provides automated, context-aware responses for hotel booking inquiries.

## 🎯 Project Overview

HotelBot is an AI-powered virtual assistant that automates hotel booking interactions by understanding user queries, classifying intents, and generating appropriate responses in real-time. The system reduces manual workload while providing customers with instant, accurate information about room reservations, amenities, and hotel services.

### Key Features
- 🧠 **Natural Language Processing** - Understands and processes user queries
- 🎯 **Intent Classification** - Accurately identifies user intentions using neural networks
- 💬 **Real-time Responses** - Provides instant, context-aware replies
- 📱 **Console Interface** - Simple text-based interaction
- 🔄 **Offline Functionality** - Works without internet connection
- 🎛️ **Modular Architecture** - Easy to maintain and extend

## 🏗️ System Architecture

The chatbot follows a structured pipeline:

1. **User Input** → Text queries from users
2. **Text Processing** → Tokenization and lemmatization using NLTK
3. **Intent Classification** → Neural Network (TensorFlow/Keras) classifies user intents
4. **Response Generation** → Retrieves appropriate responses from predefined dataset
5. **Chat Interface** → Displays responses via console

## 📁 Project Structure

```
HotelBot/
├── app.py                    # Main chatbot application logic
├── train.py                  # Model training and preprocessing
├── chatbot_model.keras       # Trained neural network model
├── intents.json             # Training dataset with patterns and responses
├── words.pkl                # Processed vocabulary data
├── classes.pkl              # Intent classification labels
├── CODE_OF_CONDUCT.md       # Project guidelines
└── README.md                # This file
```

## 🔧 Technology Stack

- **Programming Language:** Python 3.6+
- **Machine Learning:** TensorFlow/Keras
- **Natural Language Processing:** NLTK
- **Data Storage:** JSON & Pickle
- **Model Architecture:** Sequential Neural Network with SGD optimizer

## 📋 Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/sasa0383/An-AI-Chatbot-in-Python-and-Flask-main.git
cd An-AI-Chatbot-in-Python-and-Flask-main
```

2. **Install required dependencies:**
```bash
pip install tensorflow
pip install nltk
pip install numpy
pip install pickle-mixin
```

3. **Download NLTK data (if not already downloaded):**
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🎯 Usage

### Training the Model

If you need to retrain the model or modify the intents:

```bash
python train.py
```

This will:
- Process the `intents.json` file
- Create vocabulary and classes
- Train the neural network
- Save the model as `chatbot_model.keras`

### Running the Chatbot

Start the chatbot application:

```bash
python app.py
```

### Example Conversation

```
You: Hi, I want to book a room
Bot: Hello! I'd be happy to help you with your room booking. What type of room are you looking for?

You: Do you have parking available?
Bot: Yes, we offer complimentary parking for all our guests.

You: What time is check-in?
Bot: Check-in time is 3:00 PM. Early check-in may be available upon request.

You: quit
Bot: Goodbye! Have a great day!
```

## 🧠 Model Architecture

The neural network consists of:
- **Input Layer:** Bag-of-Words feature vector
- **Dense Layer 1:** 128 neurons with ReLU activation
- **Dropout Layer 1:** 0.5 rate for regularization
- **Dense Layer 2:** 64 neurons with ReLU activation
- **Dropout Layer 2:** 0.5 rate for regularization
- **Output Layer:** Softmax activation for intent classification

### Training Configuration
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Epochs:** 200
- **Batch Size:** 5
- **Loss Function:** Categorical Crossentropy

## 📊 Data Structure

### Intent Format (intents.json)
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?"]
    }
  ]
}
```

## 🔍 Key Components

### 1. Text Preprocessing (`train.py`)
- **Safe Tokenization:** Handles edge cases in text processing
- **Lemmatization:** Converts words to base forms
- **Bag-of-Words:** Creates numerical feature vectors

### 2. Intent Classification (`app.py`)
- **Model Loading:** Loads pre-trained neural network
- **Prediction:** Classifies user input into intents
- **Response Selection:** Retrieves appropriate responses

### 3. Chat Interface (`app.py`)
- **User Interaction:** Processes input and displays responses
- **Session Management:** Maintains conversation flow
- **Error Handling:** Manages unexpected inputs gracefully

## 🚀 Future Enhancements

- 🖥️ **GUI Integration:** Web-based interface using Flask
- 🌍 **Multilingual Support:** Multiple language processing
- 🧠 **Context Awareness:** Memory-based personalized interactions
- 🔗 **API Integration:** Real-time hotel database connectivity
- 📱 **Mobile App:** Native mobile application
- 🎯 **Advanced NLP:** Transformer-based models (BERT/GPT)

## 📈 Performance Optimizations (Phase 3)

Recent improvements include:
- Enhanced hyperparameter tuning
- Improved tokenization error handling
- Expanded intent dataset
- Better exception handling
- Code refactoring for maintainability

## 🧪 Testing

The chatbot has been tested for:
- Intent classification accuracy
- Response relevance
- Error handling
- Performance optimization
- User experience flow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Mostafa Shrif**
- GitHub: [@sasa0383](https://github.com/sasa0383)
- Project Link: [HotelBot Repository](https://github.com/sasa0383/An-AI-Chatbot-in-Python-and-Flask-main.git)

## 📚 References

- Jurafsky, D., & Martin, J. H. (2021). *Speech and Language Processing*. Pearson.
- Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

## 🐛 Issues & Support

If you encounter any issues or have questions:
1. Check the existing [Issues](https://github.com/sasa0383/An-AI-Chatbot-in-Python-and-Flask-main/issues)
2. Create a new issue with detailed description
3. Include error messages and system information

---

**⭐ Don't forget to star the repository if you found it helpful!**
