# Glassdoor Reviews Report

This repository contains a sentiment analysis report generated from predicted sentiments made by a model trained to analyze reviews of Cuiabá IT Companies' in Glassdoor. The model harnesses the power of BERT-based architecture, specifically BERTimbau, to predict the sentiment of reviews.

The live preview can be found at: [Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá](https://glassdoor-reviews-report.streamlit.app/).

![Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá](./img/app_screenshot.png)
⚠️ If you find the app in **sleep mode**, please hit the button **"Yes, get this app back up!"**.

![Yes, get this app back up!](https://docs.streamlit.io/images/streamlit-community-cloud/app-state-zzzz.png)

## Application Overview

### [🏠Home](🏠Home.py)
Serves as the primary access point for the application Loads the necessary data for analysis and implements the following sections:
  - **Introduction**: Provides an overview of the report.
  - **Positive and Negative Reviews Ranking**: Analyzes and ranks reviews based on sentiment.
  - **Company Analysis**: Evaluates company reputation.
  - **Sentiment Reviews Over Time**: Tracks sentiment trends across different time periods.
  - **Rating Star Analysis**: Examines distribution and patterns of rating stars.
  - **Employee Role Analysis**: Investigates reviews based on employee roles.
  - **Word Cloud Analysis**: Visual representation of frequently used words in reviews.
  - **Most Common Words Analysis**: Lists the most common words in reviews.
  - **N-Gram Analysis**: Analyzes most frequent sequences of words.

### [🥇Ranking geral de avaliações](./pages/1_🥇_Ranking%20geral%20de%20avaliações.py)
Shows the number of reviews and the associated sentiment for each company, ordered by the difference between positive and negative reviews.

### [📉 Avaliações ao longo do tempo](./pages/2_📉_Avaliações%20ao%20longo%20do%20tempo.py)
Presents a sentiment analysis of reviews along the time by company.

### [⭐ Avaliações por quantidade de estrelas](./pages/3_⭐_Avaliações%20por%20quantidade%20de%20estrelas.py)
Examines how reviews correlate with star ratings for a chosen company.

### [🧑‍💼 Avaliações por grupo de funcionários](./pages/4_🧑‍💼_Avaliações%20por%20grupo%20de%20funcionários.py)
Shows reviews categorized by different employee groups for a specific company.

### [☁️ Word Cloud](./pages/5_☁️_Nuvem%20de%20palavras%20por%20empresa.py)
Shows Word Cloud by company.

### [📊 Top 10 palavras mais usadas](./pages/6_📊_Top%2010%20palavras%20mais%20usadas.py)
Shows Top 10 frequent words by company.

### [🔠 NGrams](./pages/7_🔠_NGrams.py)
Shows N-Grams by company.

### [🧠 Treinamento do Modelo.py](./pages/8_🧠_Treinamento%20do%20Modelo.py)
Shows Data Preparation, Model Architecture, Model Training and Model Evaluation.

## How to Run the Application

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/glassdoor-reviews-report.git
    cd glassdoor-reviews-report
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```sh
    streamlit run 🏠Home.py
    ```
