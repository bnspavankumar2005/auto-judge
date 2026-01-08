# AlgoDifficulty AI: Algorithmic Complexity Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)

## 📝 Overview
AlgoDifficulty AI is a specialized Machine Learning framework designed to quantify the complexity of competitive programming problems. By utilizing Natural Language Processing (NLP) and supervised learning, the system automates the categorization of technical problem statements into standardized difficulty tiers (Easy, Medium, Hard) and predicts a continuous difficulty score.



## Technical Architecture

### 1. Hybrid Feature Engineering
The system employs a dual-stream feature extraction strategy to capture both semantic meaning and structural complexity:
* **NLP Vectorization**: A `TfidfVectorizer` (N-gram range 1-2) converts raw text into a 2,000-dimensional sparse matrix, capturing key terminology and phrasing.
* **Heuristic Features**: Custom extractors calculate:
    * **Keyword Density**: Occurrences of complex algorithmic domains (DP, Graphs, Trees, etc.).
    * **Math-Symbol Density**: Frequency of mathematical operators as a proxy for problem logic depth.
    * **Structural Metrics**: Character count and text length as indicators of description granularity.

### 2. Model Benchmarking & Selection
We evaluated several architectures to balance interpretability with predictive power:
* **Classification**: Logistic Regression, Linear SVM, XGBoost, and Random Forest.
* **Regression**: Ridge Regression and Random Forest Regressor.

**Winner**: **Random Forest** was selected for production due to its ability to handle non-linear feature interactions (e.g., the intersection of specific keywords with large numerical constraints).



## 📊 Performance Metrics

| Metric | Result | Insight |
| :--- | :--- | :--- |
| **Accuracy** | **56%** | High performance given the subjective nature of difficulty. |
| **Hard Class Recall** | **82%** | Exceptional ability to identify complex problems correctly. |
| **Mean Absolute Error (MAE)** | **1.68** | Predicted scores deviate by less than 1.7 points on average. |

## 🚀 Deployment & Usage

### Local Execution
1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/yourusername/algodifficulty-ai.git](https://github.com/yourusername/algodifficulty-ai.git)
    cd algodifficulty-ai
    ```
2.  **Install Environment**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Web Interface**:
    ```bash
    streamlit run app.py
    ```

### Repository Structure
```text
├── app.py                # Production Streamlit UI & Inference Pipeline
├── training_logic.ipynb  # Documented Research & Development phase
├── best_classifier.pkl   # Serialized Random Forest Classifier
├── best_regressor.pkl    # Serialized Random Forest Regressor
├── tfidf_vectorizer.pkl  # Serialized TF-IDF Processor
└── requirements.txt      # Dependency specification
