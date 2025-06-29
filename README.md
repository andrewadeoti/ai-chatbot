# Food Chatbot

A chatbot that answers food questions, shares recipes, checks food images, and shows how advanced AI works.

## Overview

This chatbot integrates multiple components to provide a rich conversational experience centered around food. It can:
- Answer questions about various dishes and cuisines
- Show images of food items
- Analyze and classify food images using computer vision
- Provide step-by-step recipes
- Perform logical reasoning about food
- Demonstrate advanced AI algorithms

## Features

### Food Information & Recipes
- Get detailed information about dishes from around the world
- Access step-by-step cooking instructions
- Learn about cultural origins and preparation methods
- Find nutritional information and dietary considerations

### Image Analysis
- Display food images from the built-in database
- Classify food images using a pre-trained neural network
- Analyze images using Azure Computer Vision
- Identify ingredients and objects in food photos

### Advanced AI Capabilities
- Natural Language Processing for understanding user queries
- Logical reasoning about food relationships
- Transformer model demonstrations
- Algorithm demonstrations (BFS, TSP, Linear Regression)

### External Integration
- Wikipedia integration for additional food information
- Azure Computer Vision API for advanced image analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/andrewadeoti/ai-chatbot.git
cd ai-chatbot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. **Important**: Train the neural network model (required for image classification):
```bash
python setup_model.py
```
*Note: The pre-trained model file (`classical_nn_model.h5`) is excluded from git due to size limits (76MB). The setup script will train a new model using the provided images.*

4. (Optional) Set up Azure Computer Vision for enhanced image analysis:
   - Set environment variables:
     - `AZURE_CV_KEY`: Your Azure Computer Vision API key
     - `AZURE_CV_ENDPOINT`: Your Azure Computer Vision endpoint

## Usage

Run the chatbot:
```bash
python main.py
```

## Available Commands & Prompts

### Basic Interaction
- **Greetings**: `hello`, `hi`, `hey`
- **Farewell**: `bye`, `exit`, `quit`
- **Help**: `show commands`
- **Thanks**: `thank you`, `thanks`

### Food Information Queries
- **General info**: `tell me about [dish name]`
- **Origin**: `where does [dish] come from?`
- **Ingredients**: `what are the main ingredients in [dish]?`
- **Taste**: `what does [dish] taste like?`
- **Health**: `is [dish] healthy?`
- **Serving**: `how is [dish] served?`

### Recipe & Preparation
- **Step-by-step**: `what are the steps to preparing [dish]?`
- **How to make**: `how do i make [dish]?`
- **Process**: `what is the process for preparing [dish]?`
- **Preparation**: `how is [dish] prepared?`

### Image Features
- **Show image**: `show me a picture of [dish]`
- **Display image**: `display an image of [dish]`
- **Image classification**: `what is in this picture of [dish]?`
- **Image analysis**: `analyze this image of [dish]`
- **Identify external image**: `identify image [file_path]`

### Advanced Features
- **Logical reasoning**: `check logic: [your logical query]`
- **AI demonstrations**:
  - `demo transformer` - Transformer model demonstration
  - `demo bfs` - Breadth-First Search algorithm
  - `demo tsp` - Traveling Salesman Problem
  - `demo regression` - Linear Regression demonstration
- **Model training**: `train model [epochs]` (default: 3)
- **Save model**: `save model`

## Example Conversations

### Getting Recipe Information
```
User: what are the steps to preparing apple pie?
Chatbot: Here are the steps to prepare Apple Pie:
Preparation time: 90 minutes
Step 1: Roll out pie crust.
Step 2: Slice apples and mix with sugar and cinnamon.
Step 3: Fill crust and top with another layer.
Step 4: Bake until crust is golden.
```

### Image Analysis
```
User: show me a picture of lasagna
Chatbot: Here is a picture of lasagna.
[Image displays]

User: what is in this picture of lasagna?
Chatbot: I think this is a picture of:
1: lasagna (85.2%)
2: pasta (12.1%)
3: food (2.7%)
```

### Food Information
```
User: tell me about pad thai
Chatbot: Pad Thai is a stir-fried noodle dish from Thailand, made with rice noodles, eggs, tofu or shrimp, and peanuts.
```

## Project Structure

```
chatbot/
├── main.py                          # Main application entry point
├── setup_model.py                   # Model training setup script
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
├── food_qa_expanded.csv            # Q&A database
├── dish_database.json              # Dish information and recipes
├── food_logical_kb.csv             # Logical reasoning knowledge base
├── chatbot_components/             # Core chatbot modules
│   ├── data_manager.py             # Data loading and management
│   ├── nlp_handler.py              # Natural language processing
│   ├── logic_engine.py             # Logical reasoning engine
│   ├── picture_system.py           # Image handling system
│   ├── vision_system.py            # Computer vision system
│   └── advanced_algorithms.py      # AI algorithm demonstrations
└── images/                         # Food image database
    ├── apple_pie/
    ├── lasagna/
    ├── pad_thai/
    └── [other dish folders]
```

**Note**: The `classical_nn_model.h5` file and `.venv/` directory are excluded from git due to size constraints.

## Technical Details

### Components
- **DataManager**: Handles loading and managing all data sources
- **NlpHandler**: Processes natural language queries using similarity matching
- **LogicEngine**: Performs logical reasoning about food relationships
- **PictureSystem**: Manages image display, classification, and analysis
- **AdvancedAlgorithms**: Demonstrates various AI algorithms

### Data Sources
- **Q&A Database**: 170+ food-related questions and answers
- **Dish Database**: 20+ dishes with recipes, cultural info, and prep times
- **Image Database**: 200+ food images across 20+ categories
- **Logical Knowledge Base**: Food relationship data for reasoning

### AI/ML Features
- **Image Classification**: CNN for food recognition (trained locally)
- **NLP**: Similarity-based question matching
- **Computer Vision**: Azure integration for advanced image analysis
- **Algorithm Demonstrations**: Educational AI algorithm examples

## Troubleshooting

### Model Training Issues
If you encounter issues with model training:
1. Ensure TensorFlow is installed: `pip install tensorflow`
2. Check that the `images/` directory contains food category subdirectories
3. Try reducing epochs: `python setup_model.py` (uses default 3 epochs)

### Missing Dependencies
If you get import errors:
```bash
pip install -r requirements.txt
```

### Large File Exclusions
The following files are excluded from git due to size limits:
- `.venv/` - Virtual environment (can be recreated)
- `classical_nn_model.h5` - Neural network model (can be retrained)
- `__pycache__/` - Python cache files
