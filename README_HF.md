# 🍽️ AI Food Chatbot - Hugging Face Spaces Deployment

This is an intelligent food chatbot that can help you with food recommendations, recipes, cultural knowledge, and image analysis.

## 🚀 Features

- **Food Recommendations**: Get personalized dish suggestions
- **Recipe Information**: Cooking instructions, prep times, ingredients
- **Cultural Knowledge**: Dish origins and history
- **Food Identification**: Analyze food images with AI
- **Wikipedia Integration**: Learn about ingredients and food history
- **Image Display**: View pictures of various dishes

## 🛠️ Technology Stack

- **Python**: Core application logic
- **Streamlit**: Web interface
- **NLTK**: Natural language processing
- **TensorFlow**: Machine learning capabilities
- **Azure Computer Vision**: Image analysis (optional)
- **Wikipedia API**: Knowledge integration

## 📋 Requirements

All dependencies are listed in `requirements.txt`:
- streamlit>=1.28.0
- numpy>=1.19.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- nltk>=3.6.0
- wikipedia>=1.4.0
- matplotlib>=3.3.0
- Pillow>=8.0.0
- tensorflow>=2.8.0
- requests>=2.25.0
- scipy>=1.7.0
- azure-cognitiveservices-vision-computervision>=0.9.0
- msrest>=0.6.0

## 🎯 How to Use

1. **Initialize**: Click "Initialize Chatbot" in the sidebar
2. **Ask Questions**: Type your food-related questions
3. **Get Recommendations**: Ask for dish suggestions
4. **Learn Recipes**: Request cooking instructions
5. **Explore Cultures**: Learn about different cuisines

## 💡 Example Commands

- "Recommend me something to eat"
- "What should I cook today?"
- "Tell me about pizza"
- "Show me a picture of sushi"
- "What is in this picture of pasta?"
- "How do I make lasagna?"
- "What's the history of sushi?"

## 🔧 Deployment

This app is configured for Hugging Face Spaces deployment with:
- **Main file**: `app.py`
- **Framework**: Streamlit
- **Python version**: 3.9+

## 📁 Project Structure

```
ai-chatbot/
├── app.py                 # Streamlit web interface
├── main.py               # Core chatbot logic
├── requirements.txt      # Python dependencies
├── chatbot_components/   # Modular components
├── images/              # Food images database
├── dish_database.json   # Food information
├── food_logical_kb.csv  # Knowledge base
└── food_qa_expanded.csv # Q&A dataset
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new food items to the database
- Improving the NLP capabilities
- Enhancing the UI/UX
- Adding new features

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ by Andrew Adeoti** 