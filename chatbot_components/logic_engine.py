"""
Logic Engine for the Food Chatbot
This module handles logical reasoning using NLTK's resolution prover.
"""
from nltk.sem import Expression
from nltk.inference import ResolutionProver

class LogicEngine:
    def __init__(self, nltk_kb):
        self.nltk_kb = nltk_kb
        self.read_expr = Expression.fromstring

    def check_kb_integrity(self):
        """Checks if the NLTK knowledge base is consistent."""
        if not self.nltk_kb:
            return "KB is empty, so it's consistent by default.", True
        
        # Proving False from the KB means it's inconsistent.
        prover = ResolutionProver()
        result = prover.prove(self.read_expr('FALSE'), self.nltk_kb, verbose=False)
        
        if result:
            return "Contradiction found in KB!", False
        else:
            return "KB is consistent.", True

    def handle_logical_reasoning(self, user_input):
        """Processes a logical query from the user against the KB."""
        if not self.nltk_kb:
            return "My logical knowledge base is not loaded. I cannot reason about that."

        try:
            query = self.read_expr(user_input)
            prover = ResolutionProver()
            result = prover.prove(query, self.nltk_kb)

            if result:
                return f"Yes, based on my knowledge, I can confirm that '{user_input}' is true."
            else:
                # To check for contradiction, try proving the negation of the query
                negated_query = self.read_expr(f'-({user_input})')
                is_false = prover.prove(negated_query, self.nltk_kb)
                if is_false:
                    return f"No, based on my knowledge, '{user_input}' is false."
                else:
                    return f"I am not sure about '{user_input}'. It is not explicitly stated in my knowledge base and cannot be inferred."

        except Exception as e:
            return f"I'm sorry, I couldn't understand that as a logical query. Please check the syntax. Error: {e}"

    def check_contradiction(self, statement):
        """Checks if a new statement contradicts the existing KB."""
        if not self.nltk_kb:
            return False, "KB is empty, no contradiction possible."
            
        try:
            new_expr = self.read_expr(statement)
            # Check if the negation of the new statement can be proven from the KB
            prover = ResolutionProver()
            negated_expr = self.read_expr(f'-({statement})')
            contradicts = prover.prove(negated_expr, self.nltk_kb)
            if contradicts:
                return True, f"This statement '{statement}' contradicts my existing knowledge."
            else:
                return False, "This statement does not contradict my knowledge."
        except Exception as e:
            return True, f"Could not parse the statement. Error: {e}"

    def add_statement_to_kb(self, statement):
        """Adds a new logical statement to the NLTK knowledge base."""
        try:
            expr = self.read_expr(statement)
            self.nltk_kb.append(expr)
            return True
        except:
            return False 