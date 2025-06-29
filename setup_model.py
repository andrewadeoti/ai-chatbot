#!/usr/bin/env python3
"""
Setup script for the Food Chatbot
This script trains the neural network model that was excluded from git due to size limits.
Run this script after cloning the repository to create the required model file.
"""

import os
import sys

def main():
    print("Food Chatbot Model Setup")
    print("=" * 40)
    
    # Check if images directory exists
    if not os.path.exists("images"):
        print("❌ Error: 'images' directory not found!")
        print("Please ensure you have the images directory with food categories.")
        return False
    
    # Check if required files exist
    required_files = [
        "chatbot_components/picture_system.py",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: Required file '{file_path}' not found!")
            return False
    
    print("✅ All required files found.")
    print("\nTraining the neural network model...")
    print("This may take several minutes depending on your system.")
    
    try:
        # Import and run the training
        from chatbot_components.picture_system import PictureSystem
        
        # Initialize the picture system
        ps = PictureSystem()
        
        # Train the model (default 3 epochs)
        result = ps.train_classical_nn(epochs=3)
        
        print(f"\n✅ {result}")
        print("\nModel training completed successfully!")
        print("You can now run the chatbot with: python main.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 