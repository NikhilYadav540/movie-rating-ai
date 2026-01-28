import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bollywood AI Predictor",
    page_icon="🎬",
    layout="wide"
)

# --- CUSTOM CSS (STYLING) ---
# This forces the background image to work even if you have Dark Mode on
st.markdown("""
    <style>
    /* Force background image on the main container */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("https://wallpapercave.com/wp/wp6666878.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Make the sidebar slightly transparent */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.5);
    }

    /* Style the main content box */
    .main-box {
        background-color: rgba(20, 20, 20, 0.9);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #FFA500;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
        color: white;
    }
    
    /* Text Colors */
    h1, h2, h3 { color: #FFA500 !important; } /* Orange Titles */
    p, label, .stMarkdown { color: #ffffff !important; } /* White Text */
    
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
try:
    model = tf.keras.models.load_model('movie_rating_model.h5')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("⚠️ Model not found. Please run train_model.py first!")
    st.stop()

# --- SIDEBAR: INDIAN MOVIE HISTORY (RESTORED) ---
st.sidebar.title("🇮🇳 Indian Hits & Flops")
st.sidebar.write("History Reference:")
indian_movies = pd.DataFrame({
    'Movie': ['Baahubali 2', 'KGF 2', 'Dangal', 'Adipurush', 'Thugs of Hindostan'],
    'Budget (Cr)': [250, 100, 70, 500, 300],
    'Verdict': ['Blockbuster', 'Blockbuster', 'Blockbuster', 'Flop', 'Flop'],
    'Rating': [8.2, 8.4, 8.3, 3.8, 4.0]
})
st.sidebar.table(indian_movies)

# --- MAIN APP UI ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2: 
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.title("🎬 Bollywood AI Predictor")
    st.write("Predict the success of your next Indian Movie!")
    st.markdown("---")
    
    # --- INPUTS ---
    
    # 1. Budget Input in CRORES
    budget_cr = st.number_input("💰 Budget (in Crores ₹)", value=100, step=10)
    
    # LOGIC RESTORED: Convert Crores to approximate 'Votes' proxy for the model
    # (Since we trained on Votes/Duration, we use Budget as a proxy for Popularity/Votes)
    votes_proxy = budget_cr * 1000 
    
    # 2. Other Inputs
    duration = st.slider("⏱️ Duration (Minutes)", 90, 210, 150)
    
    # 3. Extra manual boost (Optional)
    hype_score = st.slider("🔥 Marketing Hype (1-10)", 1, 10, 5)
    
    # Adjust votes based on hype
    final_votes_input = votes_proxy * (hype_score / 5)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("PREDICT SUCCESS", type="primary"):
        
        # Prepare Data for AI [Budget(as votes), Duration, Votes]
        input_data = np.array([[final_votes_input, duration, final_votes_input]])
        
        # Scale and Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        
        rating = prediction[0][0]
        rating = min(max(rating, 1.0), 9.5) # Clip between 1-10
        
        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.subheader("🤖 AI Prediction:")
        
        # Big Text for Rating
        st.markdown(f"<h1 style='text-align: center; font-size: 60px; color: white;'>{rating:.1f}/10</h1>", unsafe_allow_html=True)
        st.progress(int(rating * 10))
        
        col_a, col_b = st.columns(2)
        with col_a:
            if rating >= 8.0:
                st.success("🌟 ALL-TIME BLOCKBUSTER!")
            elif rating >= 6.0:
                st.success("✅ HIT / SUPER HIT")
            elif rating >= 4.0:
                st.warning("⚠️ AVERAGE")
            else:
                st.error("📉 FLOP")

        with col_b:
            # Comparison Logic
            if rating > 8.0:
                st.write("🔥 Comparable to **KGF** or **Baahubali**!")
            elif rating < 5.0:
                st.write("📉 Comparable to **Adipurush**...")
            else:
                st.write("😐 Comparable to a standard masala movie.")

    st.markdown('</div>', unsafe_allow_html=True)