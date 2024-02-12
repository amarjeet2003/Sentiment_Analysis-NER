import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, TFAutoModelForTokenClassification
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

client = gspread.authorize(creds)

import nltk
nltk.download('stopwords')
nltk.download('punkt')

logging.basicConfig(filename='news_analysis.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def scrape_cnbc():
    try:
        url = "https://www.cnbc.com/search/?query=green%20hydrogen&qsearchterm=green%20hydrogen"

        print(f"Scraping CNBC website: {url}")

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(options=options)

        driver.get(url)
        lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        match = False
        while not match:
            lastCount = lenOfPage
            time.sleep(3)
            lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            if lastCount == lenOfPage:
                match = True

        headlines = driver.find_elements(By.XPATH, "//span[@class='Card-title']")
        dates = driver.find_elements(By.XPATH, "//span[@class='SearchResult-publishedDate']")

        data = []
        for i in range(len(headlines)):
            data.append({
                'Date': dates[i].text,
                'Headline': headlines[i].text,
                'Source': 'CNBC'
            })

        driver.quit()
        print("Scraping completed successfully.")

        return data

    except Exception as e:
        logging.error(f"Error in scraping CNBC website: {e}")
        return []

# Function to fetch headlines from Google News RSS feed
def fetch_google_news_rss():
    try:
        url = "https://news.google.com/rss/search?q=green%20hydrogen&hl=en-IN&gl=IN&ceid=IN:en"
        print(f"Fetching Google News RSS feed: {url}")

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'xml')

        items = soup.find_all('item')

        data = []
        for item in items:
            data.append({
                'Date': item.pubDate.text,
                'Headline': item.title.text,
                'Source': 'Google News'
            })

        print("Fetching completed successfully.")

        return data

    except Exception as e:
        logging.error(f"Error in fetching Google News RSS feed: {e}")
        return []

# Function to perform Named Entity Recognition using Hugging Face model
def get_organization_names(text, model, tokenizer):
    try:
        print("Performing Named Entity Recognition.")
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        entities = ner_pipeline(text)

        organizations = [entity['word'] for entity in entities if entity['entity_group'] == 'ORG']
        organizations = ", ".join(organizations)

        print(f"Named Entity Recognition completed. Identified organizations: {organizations}")
        return organizations

    except Exception as e:
        logging.error(f"Error in Named Entity Recognition: {e}")
        return ""

# Function to perform sentiment analysis using Hugging Face model
def get_sentiment_score(text, sentiment_pipeline):
    try:
        print("Performing Sentiment Analysis.")
        result = sentiment_pipeline(text)
        sentiment_score = result[0]['score']

        print(f"Sentiment Analysis completed. Score: {sentiment_score}")
        return sentiment_score

    except Exception as e:
        logging.error(f"Error in Sentiment Analysis: {e}")
        return 0.0

def main():
    print("Starting news analysis script.")
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = TFAutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    cnbc_data = scrape_cnbc()
    google_news_data = fetch_google_news_rss()

    combined_data = cnbc_data + google_news_data

    df = pd.DataFrame(combined_data)
    sentiment_pipeline = pipeline('sentiment-analysis')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

    df['Sentiment Score'] = df['Headline'].apply(get_sentiment_score, sentiment_pipeline=sentiment_pipeline)

    df['Organization Names'] = df['Headline'].apply(get_organization_names, model=model, tokenizer=tokenizer)

    df['Date'] = df['Date'].astype(str)

    sheet_name = 'NewsAnalysisResults'
    sheet = client.create(sheet_name)
    worksheet = sheet.get_worksheet(0)

    worksheet.append_row(df.columns.tolist())

    worksheet.append_rows(df.values.tolist())


    sheet.share("", perm_type='anyone', role='reader')

    sheet_url = sheet.url
    print(f"Google Sheet Link: {sheet_url}")

    df.to_csv('news_analysis_results.csv', index=False)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df.set_index('Date', inplace=True)

    weekly_avg_sentiment = df.resample('W-Mon').mean()
    weekly_avg_sentiment['Week'] = weekly_avg_sentiment.index.to_series().dt.isocalendar().week


    plt.figure(figsize=(10, 6))
    plt.plot(weekly_avg_sentiment['Week'], weekly_avg_sentiment['Sentiment Score'], marker='o')
    plt.title('Week-wise Trend of Average Sentiment Scores')
    plt.xlabel('Week')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()

    organizations_text = ' '.join(df['Organization Names'].dropna())

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(organizations_text)
    filtered_words = [word.lower() for word in word_tokens if word.isalpha() and word.lower() not in stop_words]

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud Map of Organization Names in News Headlines')
    plt.show()

    print("News analysis completed. Results saved to 'news_analysis_results.csv'.")
    print("Exiting news analysis script.")


if __name__ == "__main__":
    main()
