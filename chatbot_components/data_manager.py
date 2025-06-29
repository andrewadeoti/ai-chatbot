"""
Data Manager for the Food Chatbot
This module handles loading all data from external files.
- Dish database from JSON
- Q&A pairs from CSV
- Logical knowledge base from CSV
"""
import json
import csv
import os
from nltk.sem import Expression

class DataManager:
    def __init__(self, image_path="images"):
        self.IMAGE_PATH = image_path
        self.dishes = {}
        self.dish_ingredients = {}
        self.kb = {}
        self.nltk_kb = []
        self.qa_database = {}
        self.qa_questions = []

    def load_all_data(self):
        """Loads all data sources for the chatbot."""
        print("Loading all data sources...")
        self.load_dish_database()
        self.load_qa_pairs()
        self.load_logical_kb()
        print("✓ All data loaded successfully.")

    def load_dish_database(self):
        """Loads the dish database from a JSON file."""
        try:
            with open('dish_database.json', 'r', encoding='utf-8') as file:
                self.dishes = json.load(file)
            for dish, data in self.dishes.items():
                if 'ingredients' in data:
                    self.dish_ingredients[dish] = [ing.lower() for ing in data['ingredients']]
            print(f"✓ Loaded {len(self.dishes)} dishes from dish_database.json")
        except (FileNotFoundError, json.JSONDecodeError):
            print("⚠ dish_database.json not found or is invalid. Using available images as a fallback.")
            self.dishes = self.get_available_dishes_from_images()

    def load_qa_pairs(self):
        """Loads question-answer pairs from a CSV file."""
        try:
            with open('food_qa_expanded.csv', 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) >= 2:
                        question, answer = row[0].strip(), row[1].strip()
                        self.qa_database[question] = answer
                        self.qa_questions.append(question)
            print(f"✓ Loaded {len(self.qa_questions)} Q/A pairs from food_qa_expanded.csv")
        except FileNotFoundError:
            print("⚠ food_qa_expanded.csv not found. Q/A functionality will be limited.")

    def load_logical_kb(self):
        """Loads the logical knowledge base from a CSV file."""
        try:
            read_expr = Expression.fromstring
            with open('food_logical_kb.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:
                        statement = row[0].strip()
                        self.kb[statement] = True
                        self.nltk_kb.append(read_expr(statement))
            print(f"✓ Loaded {len(self.nltk_kb)} logical statements from food_logical_kb.csv")
        except FileNotFoundError:
            print("⚠ food_logical_kb.csv not found. Logical reasoning will be limited.")

    def get_available_dishes_from_images(self):
        """Generates a basic dish list from the image folder structure."""
        dishes = {}
        if os.path.exists(self.IMAGE_PATH):
            for dish_folder in os.listdir(self.IMAGE_PATH):
                if os.path.isdir(os.path.join(self.IMAGE_PATH, dish_folder)):
                    dishes[dish_folder.replace('_', ' ')] = {
                        "culture": "Unknown",
                        "type": "Unknown",
                        "description": "A delicious dish."
                    }
        return dishes 