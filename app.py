import streamlit as st
import os
import sys
from datetime import datetime
import time

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the chatbot components
try:
    from main import FoodChatbot
except ImportError as e:
    st.error(f"Error importing chatbot: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Food Chatbot",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_chatbot():
    """Initialize the chatbot with a loading spinner"""
    with st.spinner("Initializing AI Food Chatbot..."):
        try:
            st.session_state.chatbot = FoodChatbot()
            st.session_state.initialized = True
            st.success("Chatbot initialized successfully!")
            time.sleep(1)
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.stop()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ AI Food Chatbot</h1>
        <p>Your intelligent food assistant - Ask about recipes, get recommendations, and explore cuisines!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("🤖 About This Chatbot")
        st.markdown("""
        This AI chatbot can help you with:
        
        🍳 **Food Recommendations**
        - Get personalized dish suggestions
        - Explore different cuisines
        
        📖 **Recipe Information**
        - Cooking instructions
        - Preparation times
        - Ingredient lists
        
        🌍 **Cultural Knowledge**
        - Dish origins and history
        - Cultural significance
        
        🔍 **Food Identification**
        - Analyze food images
        - Get detailed descriptions
        
        📚 **Wikipedia Integration**
        - Learn about ingredients
        - Discover food history
        """)
        
        st.header("💡 Try These Commands")
        st.markdown("""
        - "Recommend me something to eat"
        - "What should I cook today?"
        - "Tell me about pizza"
        - "Show me a picture of sushi"
        - "What is in this picture of pasta?"
        - "How do I make lasagna?"
        """)
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Chatbot"):
                initialize_chatbot()
        else:
            st.success("✅ Chatbot Ready!")
            
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.initialized:
        st.info("👈 Click 'Initialize Chatbot' in the sidebar to start!")
        return
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 AI:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    # Text input
    user_input = st.text_input(
        "💬 Ask me about food, recipes, or get recommendations:",
        placeholder="e.g., 'Recommend me something to eat' or 'Tell me about pizza'",
        key="user_input"
    )
    
    # Send button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        send_button = st.button("🚀 Send", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Get bot response
        with st.spinner("🤖 Thinking..."):
            try:
                response = st.session_state.chatbot.process_input(user_input)
                # Clean up the response (remove "chatbot: " prefix if present)
                if response.startswith("chatbot: "):
                    response = response[9:]
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
            except Exception as e:
                error_response = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_response,
                    "timestamp": datetime.now()
                })
        
        # Clear the input
        st.rerun()
    
    # Handle Enter key
    if user_input and not send_button:
        # This will be handled by the button click above
        pass

if __name__ == "__main__":
    main() 