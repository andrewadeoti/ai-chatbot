"""
Food Chatbot - Main Application
This is the main entry point for running the food chatbot.
It integrates all the components from the `chatbot_components` directory.
"""
import os
import re
import random
import wikipedia
from datetime import datetime
import argparse

# Import the refactored components
from chatbot_components.data_manager import DataManager
from chatbot_components.nlp_handler import NlpHandler
from chatbot_components.logic_engine import LogicEngine
from chatbot_components.picture_system import PictureSystem
from chatbot_components.advanced_algorithms import AdvancedAlgorithms

# Download NLTK data if not present
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except (ImportError, LookupError):
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded.")

class FoodChatbot:
    """The main class for the Food Chatbot, orchestrating all components."""
    def __init__(self):
        # I initialize the chatbot and all its components here
        print("Initializing Food Chatbot...")
        azure_credentials = (
            os.environ.get("AZURE_CV_KEY"), 
            os.environ.get("AZURE_CV_ENDPOINT")
        )
        if not all(azure_credentials):
            print("Azure credentials not found in environment variables.")
            print("Azure Computer Vision features will be disabled. To enable, set AZURE_CV_KEY and AZURE_CV_ENDPOINT.")
            azure_credentials = None
        # I load all the data and set up the main components
        self.data_manager = DataManager()
        self.data_manager.load_all_data()
        self.nlp_handler = NlpHandler(self.data_manager.qa_questions)
        self.logic_engine = LogicEngine(self.data_manager.nltk_kb)
        self.picture_system = PictureSystem(azure_credentials=azure_credentials)
        self.advanced_algorithms = AdvancedAlgorithms()
        # I keep track of the user's name and chatbot state
        self.user_name = "User"
        self.is_running = True

    def greet(self):
        """I generate a greeting message."""
        greetings = ["Hello! What can I help you with?", "Hi there! How can I assist you today?"]
        return f"chatbot: {random.choice(greetings)}"

    def farewell(self):
        """I generate a farewell message and stop the bot."""
        self.is_running = False
        return "chatbot: Goodbye! Have a great day."
        
    def get_info_on_dish(self, dish_name):
        """I provide information about a specific dish from the database."""
        dish_info = self.data_manager.dishes.get(dish_name.lower())
        if dish_info:
            return f"chatbot: {dish_name.title()} is a {dish_info.get('type', 'dish')} from {dish_info.get('culture', 'an unknown culture')}. {dish_info.get('description', '')}"
        return f"chatbot: I don't have information about {dish_name}."

    def get_wikipedia_summary(self, query):
        """I fetch a summary from Wikipedia for a given query."""
        try:
            return f"chatbot: Here's what I found on Wikipedia about {query}:\n{wikipedia.summary(query, sentences=3)}"
        except wikipedia.exceptions.PageError:
            return f"chatbot: Sorry, I couldn't find a Wikipedia page for '{query}'."
        except wikipedia.exceptions.DisambiguationError as e:
            return f"chatbot: '{query}' could refer to many things. Can you be more specific? Options: {e.options[:3]}"
        except Exception as e:
            return f"chatbot: An error occurred while searching Wikipedia: {e}"

    def process_input(self, user_input):
        """This is the main logic for processing user input and routing to the right feature."""
        if not user_input:
            return ""
        lower_input = user_input.lower()
        # I handle greetings, farewells, and basic commands
        if re.search(r'^\b(hello|hi|hey)\b', lower_input):
            return self.greet()
        if re.search(r'^\b(bye|exit|quit)\b', lower_input):
            return self.farewell()
        if re.search(r'^\b(thank you|thanks)\b', lower_input):
            return "chatbot: You're welcome!"
        if lower_input == 'show commands':
            return self.show_commands()
        # I handle image display with a variety of prompts
        image_display_patterns = [
            r'show me (a picture of |an image of |a |the )?(.+)',
            r'display (an image of |a picture of |a |the )?(.+)',
            r'show (an image of |a picture of |a |the )?(.+)',
            r'display image of (.+)',
            r'display (.+)',
            r'show (.+)'
        ]
        for pattern in image_display_patterns:
            match = re.match(pattern, user_input, re.IGNORECASE)
            if match:
                dish_name = match.groups()[-1].lower().strip()
                image_path, error = self.picture_system.get_random_image(dish_name)
                if error:
                    return f"chatbot: {error}"
                self.picture_system.display_image(image_path)
                return f"chatbot: Here is a picture of {dish_name}."
        # I handle image classification and Azure CV analysis
        match = re.match(r'what is in this picture of (.+)\??', user_input, re.IGNORECASE)
        if match:
            dish_name = match.group(1).lower().strip()
            image_path, error = self.picture_system.get_random_image(dish_name)
            if error:
                return f"chatbot: {error}"
            return f"chatbot: {self.picture_system.classify_food_image(image_path)}"
        
        # Handle food recommendation commands
        food_recommendation_patterns = [
            r'show me food',
            r'recommend me something',
            r'show me something to eat',
            r'recommend me food to eat',
            r'what should i eat',
            r'what can i cook',
            r'suggest a dish',
            r'random food',
            r'random dish',
            r'show me something else',
            r'recommend me something else',
            r'show something else',
            r'recommend something else',
            r'what else can i eat',
            r'what else should i cook',
            r'suggest something else',
            r'another dish',
            r'different food',
            r'show me another dish',
            r'recommend another dish'
        ]
        for pattern in food_recommendation_patterns:
            if re.match(pattern, user_input, re.IGNORECASE):
                # Get a random dish from the database
                available_dishes = list(self.data_manager.dishes.keys())
                if not available_dishes:
                    return "chatbot: Sorry, I don't have any dishes in my database."
                
                random_dish = random.choice(available_dishes)
                dish_info = self.data_manager.dishes[random_dish]
                
                # Get a random image of the dish
                image_path, error = self.picture_system.get_random_image(random_dish)
                
                # Build the response
                response = f"chatbot: I recommend trying {random_dish.title()}!\n\n"
                
                # Add dish description
                if 'description' in dish_info:
                    response += f"Description: {dish_info['description']}\n\n"
                
                # Add culture/origin info
                if 'culture' in dish_info:
                    response += f"Origin: {dish_info['culture']}\n\n"
                
                # Add recipe if available
                if 'recipe' in dish_info:
                    response += "How to make it:\n"
                    steps = dish_info['recipe']
                    for i, step in enumerate(steps, 1):
                        response += f"Step {i}: {step}\n"
                    
                    # Add prep time if available
                    if 'prep_time' in dish_info:
                        response += f"\nPreparation time: {dish_info['prep_time']} minutes\n"
                
                # Add image
                if image_path and not error:
                    response += f"\nHere's what {random_dish.title()} looks like:\n"
                    self.picture_system.display_image(image_path)
                elif error:
                    response += f"\n(I don't have an image for {random_dish.title()})"
                
                return response
        
        # Handle "identify image" command with more flexible pattern matching
        identify_patterns = [
            r'identify image (.+)',
            r'identify this image (.+)',
            r'identify this (.+)',
            r'what is this image (.+)',
            r'classify image (.+)',
            r'analyze image (.+)'
        ]
        for pattern in identify_patterns:
            match = re.match(pattern, user_input, re.IGNORECASE)
            if match:
                print(f"DEBUG: Matched pattern '{pattern}' with input '{user_input}'")
                image_path = match.group(1).strip()
                print(f"DEBUG: Extracted path: '{image_path}'")
                # Handle Windows paths with backslashes
                image_path = image_path.replace('\\', '/')
                print(f"DEBUG: Normalized path: '{image_path}'")
                
                # Check if file exists and is an image
                if not os.path.exists(image_path):
                    return f"chatbot: File not found: {image_path}"
                if not re.search(r'\.(jpg|jpeg|png|bmp|gif)$', image_path, re.IGNORECASE):
                    return "chatbot: Please provide a valid path to an image file (e.g., C:/Users/user/image.jpg)."
                
                try:
                    classification_result = self.picture_system.classify_food_image(image_path)
                    guess = self.guess_dish_from_label(classification_result)
                    if guess:
                        return f"chatbot: {classification_result}\nMy guess is that this dish is: {guess}."
                    else:
                        return f"chatbot: {classification_result}"
                except Exception as e:
                    return f"chatbot: Error processing image: {e}"
        
        match = re.match(r'analyze this image of (.+)', user_input, re.IGNORECASE)
        if match:
            dish_name = match.group(1).lower().strip()
            image_path, error = self.picture_system.get_random_image(dish_name)
            if error:
                return f"chatbot: {error}"
            return f"chatbot: {self.picture_system.analyze_image_with_azure_cv(image_path)}"
        # I handle logical reasoning queries
        if lower_input.startswith("check logic:"):
            query = user_input[len("check logic:"):].strip()
            return f"chatbot: {self.logic_engine.handle_logical_reasoning(query)}"
        
        # I handle nationality questions
        nationality_patterns = [
            r'is (.+) (japanese|chinese|korean|thai|indian|italian|french|spanish|american|british|greek|turkish|peruvian|mexican)\??',
            r'is (.+) from (japan|china|korea|thailand|india|italy|france|spain|usa|america|uk|britain|greece|turkey|peru|mexico)\??',
            r'is (.+) a (japanese|chinese|korean|thai|indian|italian|french|spanish|american|british|greek|turkish|peruvian|mexican) dish\??',
            r'what nationality is (.+)\??',
            r'where is (.+) from\??',
            r'what country is (.+) from\??'
        ]
        
        for pattern in nationality_patterns:
            match = re.match(pattern, user_input, re.IGNORECASE)
            if match:
                dish_name = match.group(1).strip().lower().replace(' ', '_')
                nationality = match.group(2).strip().lower()
                
                # Get dish info from database
                dish_info = self.data_manager.dishes.get(dish_name)
                if not dish_info:
                    return f"chatbot: I don't have information about {dish_name.replace('_', ' ')}."
                
                culture = dish_info.get('culture', '').lower()
                
                # Map nationality terms to culture terms
                nationality_map = {
                    'japanese': 'japan',
                    'chinese': 'china', 
                    'korean': 'korea',
                    'thai': 'thailand',
                    'indian': 'india',
                    'italian': 'italy',
                    'french': 'france',
                    'spanish': 'spain',
                    'american': 'usa',
                    'british': 'uk',
                    'greek': 'greece',
                    'turkish': 'turkey',
                    'peruvian': 'peru',
                    'mexican': 'mexico'
                }
                
                # Check if the nationality matches the culture
                is_match = False
                if nationality in nationality_map:
                    is_match = nationality_map[nationality] in culture
                elif nationality in ['usa', 'america']:
                    is_match = 'usa' in culture
                elif nationality in ['uk', 'britain']:
                    is_match = 'uk' in culture
                
                # Format the response
                if 'is' in user_input.lower() and '?' in user_input:
                    # Yes/No question
                    answer = "Yes" if is_match else "No"
                    return f"chatbot: {answer}. {dish_name.replace('_', ' ').title()} is from {culture.title()}."
                else:
                    # Information question
                    return f"chatbot: {dish_name.replace('_', ' ').title()} is from {culture.title()}."
        
        # I handle Wikipedia and database info queries
        if lower_input.startswith("tell me about"):
            query = user_input[len("tell me about"):].strip()
            if query in self.data_manager.dishes:
                return self.get_info_on_dish(query)
            return self.get_wikipedia_summary(query)
        # I handle AI demonstration commands
        if lower_input == 'demo transformer':
            return f"chatbot: {self.advanced_algorithms.demonstrate_transformer()}"
        if lower_input == 'demo bfs':
            return f"chatbot: {self.advanced_algorithms.demonstrate_bfs()}"
        if lower_input == 'demo tsp':
            return f"chatbot: {self.advanced_algorithms.demonstrate_tsp()}"
        if lower_input == 'demo regression':
            return f"chatbot: {self.advanced_algorithms.demonstrate_linear_regression()}"
        # I handle model training and saving
        if lower_input.startswith("train model"):
            try:
                epochs = int(user_input.split()[-1])
            except (ValueError, IndexError):
                epochs = 3
            return f"chatbot: {self.picture_system.train_classical_nn(epochs=epochs)}"
        if lower_input == "save model":
            return f"chatbot: {self.picture_system.save_model()}"
        # I handle a wide variety of preparation queries
        prep_patterns = [
            r'what are the steps to preparing (.+)',
            r'how do i make (.+)',
            r'how to make (.+)',
            r'what is the process for preparing (.+)',
            r'what are the procedures involved in preparing (.+)',
            r'how is (.+) prepared',
            r'what preparation is required for (.+)',
            r'what does the preparation for (.+) entail',
            r'how should one go about preparing (.+)',
            r'how do you get ready for (.+)',
            r'how do you prepare for (.+)',
            r'what do you need to do to prepare (.+)',
            r"what's the prep like for (.+)",
            r'what goes into getting ready for (.+)',
            r'what is the step-by-step guide to preparing (.+)',
            r'what are the key stages in preparing (.+)',
            r'can you outline the steps to prepare (.+)',
            r"what's the best way to go about preparing (.+)"
        ]
        for pattern in prep_patterns:
            match = re.match(pattern, user_input, re.IGNORECASE)
            if match:
                dish_name = match.group(1).strip().lower()
                dish_info = self.data_manager.dishes.get(dish_name)
                if dish_info and 'recipe' in dish_info:
                    steps = dish_info['recipe']
                    prep_time = dish_info.get('prep_time')
                    prep_time_str = f"Preparation time: {prep_time} minutes\n" if prep_time else ""
                    steps_str = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
                    return f"chatbot: Here are the steps to prepare {dish_name.title()}:\n{prep_time_str}{steps_str}"
                elif dish_info and 'description' in dish_info:
                    return f"chatbot: {dish_info['description']}"
                else:
                    return f"chatbot: Sorry, I don't have the preparation steps for {dish_name}."
        # I handle fallback Q&A using NLP similarity and fill in dynamic answers
        similar_question = self.nlp_handler.find_similar_question(user_input)
        if similar_question:
            question, score = similar_question
            answer = self.data_manager.qa_database.get(question, "I'm not sure how to answer that.")
            dish_match = re.search(r'\b([a-zA-Z ]+)\b', user_input)
            dish_name = None
            for dish in self.data_manager.dishes:
                if dish in user_input.lower():
                    dish_name = dish
                    break
            if '[dish]' in answer and dish_name:
                if 'main ingredients' in answer or 'made of' in question or 'ingredients' in question:
                    ingredients = self.data_manager.dish_ingredients.get(dish_name)
                    if ingredients:
                        answer = f"The main ingredients are {', '.join(ingredients)}."
                    else:
                        answer = answer.replace('[dish]', dish_name)
                elif 'associated with' in answer or 'origin' in question or 'culture' in question or 'background' in question:
                    culture = self.data_manager.dishes[dish_name].get('culture', 'Unknown')
                    answer = f"{dish_name.title()} is associated with {culture}."
                elif 'served' in answer or 'sides' in question or 'accompaniments' in question:
                    answer = answer.replace('[dish]', dish_name)
                elif 'cooked' in answer or 'cooking method' in question or 'prepared' in question:
                    desc = self.data_manager.dishes[dish_name].get('description')
                    if desc:
                        answer = desc
                    else:
                        answer = answer.replace('[dish]', dish_name)
                elif 'tastes' in answer or 'taste' in question or 'texture' in question or 'spicy' in question or 'sweet or savory' in question:
                    answer = answer.replace('[dish]', dish_name)
                elif 'healthy' in answer or 'good for you' in question or 'nutritional' in question or 'calories' in question or 'vegetarian' in question or 'vegan' in question:
                    answer = answer.replace('[dish]', dish_name)
                else:
                    answer = answer.replace('[dish]', dish_name)
            return f"chatbot: (Thinking...) I found a similar question with {score:.2%} confidence.\nQ: {question}\nA: {answer}"
        # If nothing matches, I let the user know
        return "chatbot: I'm not sure how to respond to that. You can ask me to 'show commands'."

    def show_commands(self):
        """I display a list of available commands for the user."""
        commands = """
        Here are some things you can ask me:
        - Basic Commands:
          - 'hello', 'hi', 'hey' -> Greet the chatbot.
          - 'bye', 'exit', 'quit' -> End the conversation.
          - 'show commands' -> Display this list of commands.
          - 'thank you', 'thanks' -> Respond politely.
        
        - Food Recommendations:
          - 'show me food' -> Recommends a random dish with recipe and image.
          - 'recommend me something' -> Suggests a random dish to try.
          - 'show me something to eat' -> Shows a random dish with cooking instructions.
          - 'recommend me food to eat' -> Recommends a dish with full details.
          - 'what should i eat' -> Suggests a random dish.
          - 'what can i cook' -> Shows a random dish with recipe.
          - 'suggest a dish' -> Recommends a random dish.
          - 'random food' -> Shows a random dish with image.
          - 'random dish' -> Recommends a random dish.
          - 'show me something else' -> Recommends a different random dish.
          - 'recommend me something else' -> Suggests another random dish.
          - 'show something else' -> Shows a different random dish.
          - 'recommend something else' -> Recommends another random dish.
          - 'what else can i eat' -> Suggests a different dish.
          - 'what else should i cook' -> Shows another dish to cook.
          - 'suggest something else' -> Recommends a different dish.
          - 'another dish' -> Shows a different random dish.
          - 'different food' -> Recommends a different random dish.
          - 'show me another dish' -> Shows another random dish.
          - 'recommend another dish' -> Recommends a different random dish.
        
        - Food and Information:
          - 'show me a picture of <dish_name>' -> Displays an image of a specified dish.
          - 'tell me about <topic>' -> Provides info from the database or a Wikipedia summary.
          - 'is <dish> <nationality>?' -> Answers yes/no about dish nationality (e.g., 'is pizza japanese?')
          - 'is <dish> from <country>?' -> Checks if dish is from specific country (e.g., 'is lasagna from italy?')
          - 'what nationality is <dish>?' -> Tells you the origin of a dish (e.g., 'what nationality is bibimbap?')
          - 'where is <dish> from?' -> Shows the country of origin for a dish.
        
        - Image Analysis (AI Vision):
          - 'what is in this picture of <dish_name>?' -> Classifies the main subject of an image.
          - 'analyze this image of <dish_name>' -> Uses Azure CV for object detection.
          - 'identify image <path/to/image.jpg>' -> Identifies the food in a local image file.
        
        - Logical Reasoning (Task-b):
          - 'check logic: <statement>' -> Checks a logical statement against the knowledge base.
             (e.g., 'check logic: all(x, (is_vegetarian(x) -> -is_meat(x)))')
        
        - AI Model Management:
          - 'train model <epochs>' -> Trains the internal image classifier (e.g., 'train model 5').
          - 'save model' -> Saves the trained classifier to disk.
          
        - AI Demonstrations:
          - 'demo transformer' -> Shows the transformer model architecture.
          - 'demo bfs' -> Demonstrates Breadth-First Search.
          - 'demo tsp' -> Demonstrates the Traveling Salesperson Problem solver.
          - 'demo regression' -> Demonstrates Linear Regression.
        
        - General Questions:
          - You can ask general questions, and I will try to find the most similar answer from my database.
        """
        return f"chatbot: {commands.strip()}"

    def guess_dish_from_label(self, classification_result):
        """I try to guess the dish name from the classification result."""
        match = re.search(r'1: ([^\(]+)\(', classification_result)
        if not match:
            return None
        top_label = match.group(1).strip().lower()
        for dish in self.data_manager.dishes:
            if top_label in dish.lower() or dish.lower() in top_label:
                return dish.title()
        return None

    def run(self):
        """I start the main chatbot interaction loop."""
        print("\n" + "="*50)
        print(" Food Chatbot is now running! ".center(50, "="))
        print("="*50)
        print("Type 'show commands' for a list of what I can do, or 'exit' to quit.")
        print(self.greet())
        while self.is_running:
            try:
                user_input = input(f"{self.user_name}: ").strip()
                response = self.process_input(user_input)
                if response:
                    print(response)
            except (KeyboardInterrupt, EOFError):
                print("\n" + self.farewell())
                break
        print("\nChatbot session ended.")

if __name__ == "__main__":
    # I create and run the chatbot here
    bot = FoodChatbot()
    bot.run() 