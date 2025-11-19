import streamlit as st
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #1f77b4;
    }
    .sentiment-poor { color: #ff4b4b; font-weight: bold; }
    .sentiment-fair { color: #ffa500; font-weight: bold; }
    .sentiment-neutral { color: #808080; font-weight: bold; }
    .sentiment-satisfactory { color: #32cd32; font-weight: bold; }
    .sentiment-good { color: #008000; font-weight: bold; }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff4b4b, #ffa500, #808080, #32cd32, #008000);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """Load the model and tokenizer from Hugging Face"""
    try:
        model_name = "MuhammadSaad1234/my-uploaded-checkpoint"
        
        # Show loading spinner
        with st.spinner("Loading sentiment analysis model from Hugging Face... This may take a few minutes."):
            
            # Load tokenizer
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            
            # Load model for sequence classification
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set model to evaluation mode
            model.eval()
            
            return model, tokenizer
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for the given text"""
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and probabilities
        predicted_class = torch.argmax(predictions, dim=1).item()
        probabilities = predictions.numpy()[0]
        
        return predicted_class, probabilities
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def get_sentiment_label(class_idx):
    """Convert class index to sentiment label"""
    sentiment_labels = {
        0: "Poor",
        1: "Fair", 
        2: "Neutral",
        3: "Satisfactory",
        4: "Good"
    }
    return sentiment_labels.get(class_idx, "Unknown")

def get_sentiment_color(class_idx):
    """Get CSS class for sentiment color"""
    sentiment_colors = {
        0: "sentiment-poor",
        1: "sentiment-fair",
        2: "sentiment-neutral", 
        3: "sentiment-satisfactory",
        4: "sentiment-good"
    }
    return sentiment_colors.get(class_idx, "")

def main():
    # Header
    st.markdown('<div class="main-header">😊 Sentiment Analysis App</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Model Information")
    st.sidebar.markdown("""
    **Model:** Sentiment Analysis  
    **Author:** MuhammadSaad1234  
    **Platform:** Hugging Face 🤗
    **Classes:** 5 Sentiment Levels
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Sentiment Classes")
    st.sidebar.markdown("""
    - 🟥 **Poor** (0) - Very Negative
    - 🟧 **Fair** (1) - Negative  
    - 🟩 **Neutral** (2) - Neutral
    - 🟦 **Satisfactory** (3) - Positive
    - 🟪 **Good** (4) - Very Positive
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "This model analyzes text reviews and predicts sentiment across 5 classes "
        "from Poor to Good. It's fine-tuned on women's clothing reviews data."
    )
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check if the model is properly uploaded to Hugging Face.")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Review")
        
        # Text input area
        user_input = st.text_area(
            "Type your review here:",
            height=150,
            placeholder="Enter your product review or any text for sentiment analysis...",
            help="The model will analyze the sentiment of your text and classify it into one of 5 categories."
        )
        
        # Predict button
        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    predicted_class, probabilities = predict_sentiment(user_input, model, tokenizer)
                    
                    if predicted_class is not None:
                        sentiment_label = get_sentiment_label(predicted_class)
                        sentiment_color = get_sentiment_color(predicted_class)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("Analysis Results")
                        
                        # Sentiment prediction
                        st.markdown(f"**Predicted Sentiment:** <span class='{sentiment_color}'> {sentiment_label} ({predicted_class})</span>", 
                                  unsafe_allow_html=True)
                        
                        # Confidence scores
                        st.markdown("**Confidence Scores:**")
                        
                        # Create a nice visualization of probabilities
                        for i, prob in enumerate(probabilities):
                            label = get_sentiment_label(i)
                            color_class = get_sentiment_color(i)
                            percentage = prob * 100
                            
                            col_prob1, col_prob2 = st.columns([2, 3])
                            with col_prob1:
                                st.markdown(f"<span class='{color_class}'>{label}: {percentage:.2f}%</span>", 
                                          unsafe_allow_html=True)
                            with col_prob2:
                                st.progress(float(prob))
                        
                        # Additional insights
                        st.markdown("---")
                        st.subheader("Detailed Analysis")
                        
                        max_prob_idx = np.argmax(probabilities)
                        max_prob = probabilities[max_prob_idx] * 100
                        
                        if max_prob > 80:
                            st.success(f"✅ **High Confidence**: The model is {max_prob:.1f}% confident in this prediction.")
                        elif max_prob > 60:
                            st.info(f"ℹ️ **Moderate Confidence**: The model is {max_prob:.1f}% confident in this prediction.")
                        else:
                            st.warning(f"⚠️ **Low Confidence**: The model is only {max_prob:.1f}% confident. The text might be ambiguous.")
                            
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.subheader("Example Reviews")
        st.markdown("Try these examples or create your own:")
        
        examples = [
            "This dress is absolutely amazing! Perfect fit and great quality.",
            "The product was okay, nothing special but not bad either.",
            "Terrible quality. The fabric is cheap and it arrived damaged.",
            "Pretty good for the price. Would recommend to others.",
            "Not what I expected. The color is different from the picture."
        ]
        
        for example in examples:
            if st.button(example[:50] + "..." if len(example) > 50 else example, 
                        key=example, 
                        use_container_width=True):
                st.session_state.user_input = example
                st.rerun()
        
        st.markdown("---")
        st.subheader("Model Stats")
        st.metric("Input Length", f"128 tokens max")
        st.metric("Model Type", "BERT-base")
        st.metric("Classes", "5")

if __name__ == "__main__":
    main()
