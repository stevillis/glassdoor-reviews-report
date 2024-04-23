# Glassdoor Reviews Report

This repository contains a sentiment analysis report generated from predicted sentiments made by a model trained to analyze reviews of Cuiabá IT Companies' in Glassdoor. The model harnesses the power of BERT-based architecture, specifically BERTimbau, to predict the sentiment of reviews. The development process comprised the following key steps:

### Data Extraction
Reviews were extracted from Glassdoor, a leading platform for employee reviews of companies. These reviews serve as the dataset for training the sentiment analysis model.

### Sentiment Annotation
Some reviews needed to be annotated with sentiment labels, indicating whether the sentiment expressed was positive, negative, or neutral. This annotation process was essential for supervised learning, providing labeled data for training the model.

### Model Training
Utilizing the annotated dataset, a sentiment analysis model was trained. The model is based on BERTimbau, a variant of BERT optimized for Portuguese language tasks. During training, the model learns to predict the sentiment of reviews based on their textual content.

### Sentiment Prediction
With the trained model, sentiment analysis is performed on the extracted reviews from Glassdoor. The model predicts the sentiment of each review, facilitating automated analysis at scale.

### Analysis Report
A comprehensive report is generated to analyze the predictions made by the model. This report offers insights into the sentiment distribution and highlights potential areas for improvement. It provides actionable insights for companies, especially regarding areas that require attention, particularly for current employees.
