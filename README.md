# AlgoDifficulty AI: Algorithmic Complexity Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)

## Overview
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
* **Regression**: Ridge Regression, linear regression, XGBoost regressor and Random Forest Regressor.

**Winner**: **Random Forest** was selected for production due to its ability to handle non-linear feature interactions (e.g., the intersection of specific keywords with large numerical constraints).



## Deployment & Usage

###  Project Demo
Click the image below to watch the video demonstration of the **AlgoDifficulty AI** in action:

[![AlgoDifficulty AI Demo](https://img.shields.io/badge/Demo-Watch%20Video-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/drive/folders/1WhkEIMqHTLkZ1ze-z47YvZ7iXg2EG9dv?usp=sharing)

---

### Local Execution
To get the project running locally, follow these steps:

### Prerequisites
- Python 3.8+
- Recommended packages: `scikit-learn`, `pandas`, `numpy`, `jupyter`, `streamlit`

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/yourusername/algodifficulty-ai.git](https://github.com/yourusername/algodifficulty-ai.git)
    cd algodifficulty-ai
    ```
2.  **Install Environment**:
    ```bash
    pip install -r 
    ```
3.  **Launch Web Interface**:
    ```bash
    streamlit run app.py
    ```

---

## Repository Structure
A clear overview of the key components included in this project:

| File Name | Description |
| :--- | :--- |
| **`23116023_auto_judge_model.ipynb`** | Primary Jupyter Notebook containing data exploration, model development, training, and evaluation. |
| **`AUTO_JUDGE_REPORT.pdf`** | Comprehensive project report detailing the methodology, experiments, and final results. |
| **`README.md`** | The "instruction manual" for the repository, communicating project behavior and setup. |
| **`app.py`** | The inference script used to deploy the model or provide a user interface for predictions. |
| **`best_classifier.pkl`** | Serialized best-performing classification model saved in pickle format. |
| **`best_regressor.pkl`** | Serialized best-performing regression model (if applicable) for secondary scoring. |
| **`tfidf_vectorizer.pkl`** | The saved TF-IDF vectorizer used to transform raw text into numerical features for the model. |

---
