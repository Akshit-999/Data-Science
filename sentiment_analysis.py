
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords


import re
from nltk.tokenize import sent_tokenize



# Function to extract the article title and text from a URLs given in Input.xlxs


print("Current working directory:", os.getcwd())



def extract_a_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find and remove unwanted elements
        for ele in soup(["header", "footer"]):
            ele.decompose()
        
        # Extract article title and text
        a_title = soup.find('title').text.strip()
        a_text = ""
        
        # Extract text from <div class="td-post-content tagdiv-type">
        a_div = soup.find('div', class_='td-post-content tagdiv-type')
        if a_div:
            a_text = a_div.get_text()
        return a_title, a_text
    
    except Exception:
        print(f"Error while extracting article from {url}: {Exception}")
        return None, None

# Function to save the article title and text to a text file
    
def save_article_to_file(url_id, a_title, a_text):
    if not os.path.exists("articles"):
        os.mkdir("articles")
    
    with open(f"articles/{url_id}.txt", "w", encoding="utf-8") as file:
        file.write(f"Title: {a_title}\n\n")
        file.write(a_text)

def main():
    input_file = "/Users/akshitagrawal/Desktop/datasets/blackcoffer/input.xlsx"
    df = pd.read_excel(input_file)
    
    for index, row in df.iterrows():
        url_id = row["URL_ID"]
        url = row["URL"]
        
        # Extract article title and text
        a_title, a_text = extract_a_text(url)
        
        # Check if extraction was successful
        if a_title and a_text:
            save_article_to_file(url_id, a_title, a_text)
            print(f"Article {url_id} extracted and saved successfully.")
        else:
            print(f"Failed to extract article {url_id}.")

if __name__ == "__main__":
    main()







# Function to load positive and negative dictionaries from files
def load_dicts(positive_dict_file, negative_dict_file):
    with open(positive_dict_file, 'r', encoding="ISO-8859-1") as file:
        positive_words = set(file.read().splitlines())
    with open(negative_dict_file, 'r', encoding="ISO-8859-1") as file:
        negative_words = set(file.read().splitlines())
    return positive_words, negative_words

# Function to perform sentiment analysis and calculate scores
def calculate_sentiment_scores(text, positive_words, negative_words):
    sia = SentimentIntensityAnalyzer()
    tokens = word_tokenize(text)
    
    positive_score = 0
    negative_score = 0
    
    for word in tokens:
        # Remove punctuation and convert to lowercase
        word = word.lower()
        if word.isalpha():
            # Check if the word is in the positive dictionary
            if word in positive_words:
                positive_score += 1
            # Check if the word is in the negative dictionary
            if word in negative_words:
                negative_score += 1
    
    # Calculate sentiment analysis metrics
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    
    return positive_score, negative_score, polarity_score, subjectivity_score

def main():
    input_data_file = "/Users/akshitagrawal/Desktop/datasets/blackcoffer/Output Data Structure.xlsx"
    positive_dict_file = "/Users/akshitagrawal/Desktop/datasets/blackcoffer/positive-words.txt"
    negative_dict_file = "/Users/akshitagrawal/Desktop/datasets/blackcoffer/negative-words.txt"
    articles_dir = "articles"
    
    # Load dictionaries
    positive_words, negative_words = load_dicts(positive_dict_file, negative_dict_file)

    # Read output data structure Excel file
    output_data = pd.read_excel(input_data_file)
    
    results = []
    for index, row in output_data.iterrows():
        url_id = row["URL_ID"]
        url = row["URL"]
        article_file = os.path.join(articles_dir, f"{url_id}.txt")
        
        if os.path.exists(article_file):
            # Read article text from file
            with open(article_file, 'r', encoding='utf-8') as article:
                a_text = article.read()
            
            # Perform sentiment analysis
            positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(a_text, positive_words, negative_words)
            
            results.append({
                "URL_ID": url_id,
                "URL": url,
                "Positive_Score": positive_score,
                "Negative_Score": negative_score,
                "Polarity_Score": polarity_score,
                "Subjectivity_Score": subjectivity_score
            })
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Save results to Excel
    result_df.to_excel("sentiment_analysis_results.xlsx", index=False)

if __name__ == "__main__":
    main()


sentiment_analysis = pd.read_excel("sentiment_analysis_results.xlsx")





# Function to calculate average sentence length
def calculate_avg_sentence_length(sentences):
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    return total_words / total_sentences

# Function to calculate percentage of complex words
def calculate_percentage_complex_words(text):
    words = word_tokenize(text)
    complex_words = [word for word in words if len(word) > 2]
    return len(complex_words) / len(words)

# Function to calculate fog index
def calculate_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Function to calculate average number of words per sentence
def calculate_avg_words_per_sentence(words, sentences):
    return len(words) / len(sentences)

# Function to calculate complex word count
def calculate_complex_word_count(text):
    words = word_tokenize(text)
    complex_words = [word for word in words if len(word) > 2]
    return len(complex_words)

# Function to calculate word count
def calculate_word_count(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned_words = [word for word in words if word not in stop_words and word.isalpha()]
    return len(cleaned_words)

# Function to count syllables in a word
def count_syllables(word):
    vowels = "aeiouAEIOU"
    count = 0
    if word[-1] in ['e', 'E'] and word[-2:] != 'le' and word[-2:] != 'LE':
        word = word[:-1]
    for index, letter in enumerate(word):
        if index == 0 and letter in vowels:
            count += 1
        elif letter in vowels and word[index-1] not in vowels:
            count += 1
    return count

# Function to calculate syllable count per word
def calculate_syllable_count_per_word(text):
    words = word_tokenize(text)
    syllable_count = sum(count_syllables(word) for word in words)
    return syllable_count / len(words)

# Function to calculate personal pronoun count
def calculate_personal_pronouns(text):
    pronouns = ["I", "we", "my", "ours", "us"]
    pattern = r'\b(?:' + '|'.join(pronouns) + r')\b'
    matches = re.findall(pattern, text)
    return len(matches)

# Function to calculate average word length
def calculate_avg_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    return total_characters / len(words)

def main():
    output_data_file = "/Users/akshitagrawal/Desktop/datasets/blackcoffer/Output Data Structure.xlsx"
    articles_dir = "articles"
    
    # Read output data structure Excel file
    output_data = pd.read_excel(output_data_file)
    
    results_ = []
    for index, row in output_data.iterrows():
        url_id = row["URL_ID"]
        article_file = os.path.join(articles_dir, f"{url_id}.txt")
        
        if os.path.exists(article_file):
            # Read article text from file
            with open(article_file, 'r', encoding='utf-8') as article:
                a_text = article.read()
            
            # Tokenize sentences for text analysis
            sentences = sent_tokenize(a_text)
            words = word_tokenize(a_text)
            
            # Calculate text analysis metrics
            avg_sentence_length = calculate_avg_sentence_length(sentences)
            percentage_complex_words = calculate_percentage_complex_words(a_text)
            fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)
            avg_words_per_sentence = calculate_avg_words_per_sentence(words, sentences)
            complex_word_count = calculate_complex_word_count(a_text)
            word_count = calculate_word_count(a_text)
            syllable_count_per_word = calculate_syllable_count_per_word(a_text)
            personal_pronoun_count = calculate_personal_pronouns(a_text)
            avg_word_length = calculate_avg_word_length(a_text)
            
            results_.append({
                "URL_ID": url_id,
                "Avg_Sentence_Length": avg_sentence_length,
                "Percentage_Complex_Words": percentage_complex_words,
                "Fog_Index": fog_index,
                "Avg_Words_Per_Sentence": avg_words_per_sentence,
                "Complex_Word_Count": complex_word_count,
                "Word_Count": word_count,
                "Syllable_Count_Per_Word": syllable_count_per_word,
                "Personal_Pronoun_Count": personal_pronoun_count,
                "Avg_Word_Length": avg_word_length
            })
    
    # Create DataFrame from results
    result_df2 = pd.DataFrame(results_)
    
    # Save results to Excel
    result_df2.to_excel("text_analysis_results.xlsx", index=False)

if __name__ == "__main__":
    main()


text_analysis = pd.read_excel("text_analysis_results.xlsx")


merged_df = pd.merge(sentiment_analysis, text_analysis, on='URL_ID')



merged_df.to_excel("/Users/akshitagrawal/Desktop/datasets/blackcoffer/OutputDataStructure.xlsx")


print(merged_df.columns)


