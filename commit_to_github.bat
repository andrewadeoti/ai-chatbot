@echo off
echo Food Chatbot - GitHub Commit Script
echo ===================================
echo.

echo Step 1: Initializing Git repository...
git init

echo.
echo Step 2: Adding all files (excluding .gitignore items)...
git add .

echo.
echo Step 3: Making initial commit...
git commit -m "Initial commit: Food Chatbot with AI features"

echo.
echo Step 4: Adding remote repository...
git remote add origin https://github.com/andrewadeoti/ai-chatbot.git

echo.
echo Step 5: Pushing to GitHub...
git branch -M main
git push -u origin main

echo.
echo ===================================
echo Commit completed successfully!
echo.
echo Your chatbot is now available at:
echo https://github.com/andrewadeoti/ai-chatbot.git
echo.
echo Note: The following files were excluded due to size limits:
echo - .venv/ (virtual environment)
echo - classical_nn_model.h5 (neural network model)
echo - __pycache__/ (Python cache files)
echo.
echo Users can recreate these by running:
echo - pip install -r requirements.txt (for virtual environment)
echo - python setup_model.py (for neural network model)
echo.
pause 