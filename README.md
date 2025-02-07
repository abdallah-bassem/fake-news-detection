# Fake News Detection Model

I've created a machine learning model to detect fake news using Python, implemented in a Jupyter Notebook. This project demonstrates how to build a text classification system that can distinguish between "REAL" and "FAKE" news articles with high accuracy.

[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab&style=flat-square)](https://colab.research.google.com/drive/1p2qBEpyZhikm01ylqSLBq2V9chF9-5-J?usp=sharing)
[![Open In Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-blue?logo=kaggle&style=flat-square)](https://www.kaggle.com/code/abdallahbassem369/fake-news-detection-tf-idf-logistic-regression-92)

## About the Project

I've built a complete machine learning pipeline that includes:

1. Data preprocessing and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature engineering using TF-IDF
4. Model training using Logistic Regression
5. Comprehensive evaluation metrics and visualizations

## Key Features

- Text preprocessing with NLTK, including stemming and stop word removal
- TF-IDF vectorization for feature extraction
- Logistic Regression model achieving ~92% accuracy on test data
- Detailed performance analysis with confusion matrices and classification reports
- Feature importance analysis showing top predictive words
- Efficient word processing using LRU cache
- Balanced dataset handling

## Requirements

You'll need the following Python libraries:

```bash
pip install pandas scikit-learn nltk matplotlib seaborn numpy
```

Additionally, you'll need to download the NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## Dataset

I'm using dataset from Kaggle. You can download it [here](https://www.kaggle.com/datasets/rajatkumar30/fake-news/data). Place the `news.csv` file in the project directory before running the notebook.

## How to Use

1. Clone this repository
2. Install the required dependencies
3. Download the dataset and place it in the project directory
4. Run the Jupyter Notebook `fake_news_detection.ipynb`

## Results

I've achieved the following results:
- Training accuracy: ~95%
- Testing accuracy: ~92%
- Balanced performance across both "FAKE" and "REAL" classes
- Strong cross-validation scores indicating good generalization

## Key Insights

Through my analysis, I found that:
- The dataset is perfectly balanced between real and fake news
- Fake news articles tend to be shorter than real news
- The model shows consistent performance across different evaluation metrics
