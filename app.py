import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack

def predict_difficulty(title, desc, input_desc, output_desc):
    # 1. Load the saved components
    clf = joblib.load('best_classifier.pkl')
    reg = joblib.load('best_regressor.pkl')
    vec = joblib.load('tfidf_vectorizer.pkl')
    
    # 2. Preprocess
    combined = f"{title} {desc} {input_desc} {output_desc}".lower()
    combined = re.sub(r"\s+", " ", combined).strip()
    
    # 3. Manual Feature Extraction
    length = len(combined)
    math_symbols = "+-*/=<>^"
    math_count = sum(combined.count(s) for s in math_symbols)
    
    keywords = ["graph", "dp", "recursion", "tree", "greedy" , "bitmask" , "binary search" , "geometry" , "data structures" , "brute force" ]

    kw_counts = [combined.count(kw) for kw in keywords]
    
    # Combine manual features into a 2D array
    manual_feats = np.array([[length, math_count] + kw_counts])
    
    # 4. Vectorization
    tfidf_feats = vec.transform([combined])
    
    # 5. Final Feature Matrix
    X_input = hstack([tfidf_feats, manual_feats])
    
    # 6. Predict
    class_idx = clf.predict(X_input)[0]
    score = reg.predict(X_input)[0]
    
    class_mapping = {0: "Easy", 1: "Medium", 2: "Hard"}
    return class_mapping[class_idx], round(float(score), 2)


# (Include the predict_difficulty function here or import it)

st.set_page_config(page_title="AlgoDifficulty AI", page_icon="💻")

st.title("🏆 Algorithm Difficulty Predictor")
st.markdown("Enter the details of a programming problem to predict its class and score.")

with st.form("problem_form"):
    title = st.text_input("Problem Title")
    desc = st.text_area("Description")
    col1, col2 = st.columns(2)
    with col1:
        inp = st.text_area("Input Format")
    with col2:
        out = st.text_area("Output Format")
    
    submit = st.form_submit_button("Predict Difficulty")

if submit:
    if desc.strip() == "":
        st.error("Please provide at least a problem description.")
    else:
        with st.spinner('Analyzing algorithmic complexity...'):
            p_class, p_score = predict_difficulty(title, desc, inp, out)
            
            # Display results
            st.success("Analysis Complete!")
            c1, c2 = st.columns(2)
            c1.metric("Predicted Class", p_class)
            c2.metric("Difficulty Score", p_score)