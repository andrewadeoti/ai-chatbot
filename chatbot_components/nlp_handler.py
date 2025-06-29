"""
NLP Handler for the Food Chatbot
This module handles similarity-based Q&A matching using TF-IDF and cosine similarity.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

class NlpHandler:
    def __init__(self, questions):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
        self.qa_vectors = None
        self.questions = questions
        if self.questions:
            self.prepare_similarity_search()

    def preprocess_text(self, text):
        """Lemmatizes, tokenizes, and removes stopwords from text."""
        tokens = nltk.word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]

    def prepare_similarity_search(self):
        """Creates TF-IDF vectors for the known questions."""
        if self.questions:
            self.qa_vectors = self.vectorizer.fit_transform(self.questions)
            print("✓ TF-IDF vectors created for similarity search.")

    def find_similar_question(self, user_input, threshold=0.3):
        """Finds the most similar question from the database."""
        if self.qa_vectors is None or not self.questions:
            return None

        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.qa_vectors).flatten()
        
        most_similar_idx = similarities.argmax()
        
        if similarities[most_similar_idx] >= threshold:
            return self.questions[most_similar_idx], similarities[most_similar_idx]
        return None 