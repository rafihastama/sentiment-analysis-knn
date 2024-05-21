from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import csv
import io
from io import StringIO
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
from collections import Counter
import math
import traceback
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from deep_translator import GoogleTranslator

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

translator = GoogleTranslator(source='auto', target='en')
 
app = Flask(__name__)
cors = CORS(app, origins='*')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/knn_sentiment'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

ignored_usernames = {'hariankompas', 'kompascom', 'kompastv', 'kompasbola', 'kompasmuda', 'kompasklasika', 'kompasdata'}

class Kamus(db.Model):
    __tablename__ = 'kamus'
    id = db.Column(db.Integer, primary_key=True)
    informal = db.Column(db.String(50))
    formal = db.Column(db.String(50))

class Tweet(db.Model):
    __tablename__ = 'raw-tweet-table'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.String(45))
    full_text = db.Column(db.String(800))
    username = db.Column(db.String(45))
    tweet_url = db.Column(db.String(100))
    processed_text = db.Column(db.String(500))
    sentiment = db.Column(db.DECIMAL(precision=5, scale=4))
    cleaned_text = db.Column(db.String(500))
    tokenized_words = db.Column(db.String(1000))
    formal_text = db.Column(db.String(500))
    stopword_removal = db.Column(db.String(500))

class TweetTraining(db.Model):
    __tablename__ = 'tweet-table-training'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.String(45))
    full_text = db.Column(db.String(800))
    username = db.Column(db.String(45))
    tweet_url = db.Column(db.String(100))
    processed_text = db.Column(db.String(500))
    sentiment = db.Column(db.DECIMAL(precision=5, scale=4))
    cleaned_text = db.Column(db.String(500))
    tokenized_words = db.Column(db.String(1000))
    formal_text = db.Column(db.String(500))
    stopword_removal = db.Column(db.String(500))
    
class TweetTesting(db.Model):
    __tablename__ = 'tweet-table-testing'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.String(45))
    full_text = db.Column(db.String(800))
    username = db.Column(db.String(45))
    tweet_url = db.Column(db.String(100))
    processed_text = db.Column(db.String(500))
    sentiment = db.Column(db.DECIMAL(precision=5, scale=4))
    cleaned_text = db.Column(db.String(500))
    tokenized_words = db.Column(db.String(1000))
    formal_text = db.Column(db.String(500))
    stopword_removal = db.Column(db.String(500))
    
@app.route('/import-csv', methods=['POST'])
def import_csv():
    file = request.files['file']
    if file:
        try:
            stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_reader = csv.reader(stream)
            next(csv_reader)
            for row in csv_reader:
                tweet = Tweet(created_at=row[0], full_text=row[2], username=row[10], tweet_url=row[11])
                db.session.add(tweet)
            db.session.commit()
            return jsonify({'message': 'CSV imported successfully!'}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No file provided!'}), 400
    
@app.route('/import-excel', methods=['POST'])
def import_excel():
    file = request.files['file']
    if file:
        try:
            df = pd.read_excel(file)
            for index, row in df.iterrows():
                tweet = Tweet(created_at=row['created_at'], full_text=row['full_text'], username=row['username'], tweet_url=row['tweet_url'])
                db.session.add(tweet)
            db.session.commit()
            return jsonify({'message': 'Excel imported successfully!'}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No file provided!'}), 400

@app.route('/get-data', methods=['GET'])
def get_data():
    try:
        tweets = Tweet.query.all()
        data = [{'id': tweet.id, 'created_at': tweet.created_at, 'full_text': tweet.full_text, 
                 'username': tweet.username, 'tweet_url': tweet.tweet_url, 'processed_text': tweet.processed_text, 
                 'sentiment': tweet.sentiment, 'cleaned_text': tweet.cleaned_text, 'tokenized_words': tweet.tokenized_words, 
                 'formal_text': tweet.formal_text, 'stopword_removal': tweet.stopword_removal} for tweet in tweets]
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get-both-data', methods=['GET'])
def get_both_data():
    try:
        training_count = TweetTraining.query.count()
        testing_count = TweetTesting.query.count()
        
        return jsonify({
            'training_count': training_count,
            'testing_count': testing_count
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/delete-all', methods=['DELETE'])
def delete_all_tweets():
    try:
        db.session.query(Tweet).delete()
        db.session.commit()
        return jsonify({'message': 'All tweets deleted successfully!'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@app.route('/delete-both', methods=['DELETE'])
def delete_both():
    try:
        db.session.query(TweetTraining).delete()
        db.session.query(TweetTesting).delete()
        db.session.commit()
        return jsonify({'message': 'All tweets deleted successfully!'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
def replace_informal_with_formal(word):
    informal_word = Kamus.query.filter_by(informal=word).first()
    if informal_word:
        return informal_word.formal
    else:
        return word
    
@app.route('/preprocess-tweets', methods=['POST'])
def preprocess_tweets():
    try:
        tweets_to_delete = Tweet.query.filter(Tweet.username.in_(ignored_usernames)).all()
        
        for tweet in tweets_to_delete:
            db.session.delete(tweet)
        
        tweets_to_process = Tweet.query.filter(~Tweet.username.in_(ignored_usernames)).all()
        
        # Variabel untuk menghitung ID tweet baru
        new_tweet_id = 1
        
        for tweet in tweets_to_process:
            # Cleaning
            cleaned_text = re.sub(r'@[^\s]+', '', tweet.full_text)
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
            cleaned_text = re.sub(r'&amp;', '', cleaned_text)
            cleaned_text = re.sub(r'\bamp\b', '', cleaned_text) 
            cleaned_text = re.sub(r'http\S+', '', cleaned_text)
            cleaned_text = cleaned_text.lower()
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            tweet.cleaned_text = cleaned_text
            
            # Tokenization
            tokenized_words = cleaned_text.split()
            tokenized_text_with_separator = ', '.join(tokenized_words)
            tweet.tokenized_words = tokenized_text_with_separator
            
            # Informal ke Formal
            formal_words = [replace_informal_with_formal(word) for word in tokenized_words]
            formal_text = ' '.join(formal_words)
            tweet.formal_text = formal_text
            
            # Stopword Removal
            stopword_removed_text = stopword_remover.remove(formal_text)
            tweet.stopword_removal = stopword_removed_text
            
            # Stemming
            stemmed_words = [stemmer.stem(word) for word in stopword_removed_text.split()]
            stemmed_text = ' '.join(stemmed_words)
            # translated_text = translator.translate(stemmed_text)
            tweet.processed_text = stemmed_text
            
            # Set ulang ID tweet
            tweet.id = new_tweet_id
            new_tweet_id += 1
        
        db.session.commit()
        
        return jsonify({'message': 'Tweets processed successfully!'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/label-sentiment', methods=['POST'])
def label_sentiment():
    try:
        tweet_id = request.json['tweet_id']
        sentiment_label = request.json['sentiment_label']
        
        tweet = Tweet.query.get(tweet_id)
        if tweet:
            if sentiment_label == 'Positif':
                tweet.sentiment = 1
            elif sentiment_label == 'Negatif':
                tweet.sentiment = -1
            else:
                tweet.sentiment = 0
            
            db.session.commit()
            return jsonify({'message': 'Sentiment labeled successfully!', 'sentiment': tweet.sentiment}), 200
        else:
            return jsonify({'error': 'Tweet not found!'}), 404
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@app.route('/label-sentiment-automatically', methods=['POST'])
def label_sentiment_automatically():
    try:
        tweets = Tweet.query.all()
        for tweet in tweets:
            sentiment_score = sid.polarity_scores(tweet.processed_text)['compound']
            if sentiment_score >= 0.05:
                tweet.sentiment = 1
            elif sentiment_score <= -0.05:
                tweet.sentiment = -1
            else:
                tweet.sentiment = 0
        db.session.commit()
        return jsonify({'message': 'Sentiment labeled automatically!'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@app.route('/export-data', methods=['POST'])
def export_data():
    try:
        tweets = Tweet.query.all()
        data = [
            {'id': tweet.id, 'username': tweet.username, 'full_text': tweet.full_text, 'processed_text': tweet.processed_text, 'sentiment': tweet.sentiment, 'label_sentiment': 'Netral' if tweet.sentiment == 0 else 'Negatif' if tweet.sentiment == -1 else 'Positif' if tweet.sentiment == 1 else ''}
            for tweet in tweets
        ]
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['id', 'username', 'full_text', 'processed_text', 'sentiment','label_sentiment'])
        writer.writeheader()
        writer.writerows(data)
        
        csv_data_bytes = output.getvalue().encode('utf-8')
        
        return send_file(io.BytesIO(csv_data_bytes), as_attachment=True, download_name='exported_data.csv', mimetype='text/csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/split-data', methods=['POST'])
def split_data():
    try:
        # Ambil semua data dari database
        all_tweets = Tweet.query.all()
        
        # Hitung jumlah data untuk masing-masing kategori sentimen
        sentiments = [tweet.sentiment for tweet in all_tweets]
        sentiment_counts = dict(Counter(sentiments))
        
        # Hitung jumlah data uji untuk masing-masing kategori sentimen berdasarkan persentase
        test_size_neutral = int(sentiment_counts[0] * 0.2)
        test_size_positive = int(sentiment_counts[1] * 0.2)
        test_size_negative = int(sentiment_counts[-1] * 0.2)
        
        # Bagi data menjadi data latih dan data uji dengan split stratified sampling
        train_data, test_data = [], []
        for sentiment in [-1, 0, 1]:
            sentiment_data = [tweet for tweet in all_tweets if tweet.sentiment == sentiment]
            X_train, X_test = train_test_split(sentiment_data, test_size=test_size_neutral if sentiment == 0 else
                                                                            test_size_positive if sentiment == 1 else
                                                                            test_size_negative,
                                               stratify=[tweet.sentiment for tweet in sentiment_data], random_state=42)
            train_data.extend(X_train)
            test_data.extend(X_test)
        
        # Simpan data uji ke dalam tabel TweetTesting
        for tweet in test_data:
            tweet_testing = TweetTesting(
                created_at=tweet.created_at,
                full_text=tweet.full_text,
                username=tweet.username,
                tweet_url=tweet.tweet_url,
                processed_text=tweet.processed_text,
                sentiment=tweet.sentiment,
                cleaned_text=tweet.cleaned_text,
                tokenized_words=tweet.tokenized_words,
                formal_text=tweet.formal_text,
                stopword_removal=tweet.stopword_removal
            )
            db.session.add(tweet_testing)
            
        for tweet in train_data:
            tweet_training = TweetTraining(
                created_at=tweet.created_at,
                full_text=tweet.full_text,
                username=tweet.username,
                tweet_url=tweet.tweet_url,
                processed_text=tweet.processed_text,
                sentiment=tweet.sentiment,
                cleaned_text=tweet.cleaned_text,
                tokenized_words=tweet.tokenized_words,
                formal_text=tweet.formal_text,
                stopword_removal=tweet.stopword_removal
            )
            db.session.add(tweet_training)
        
        db.session.commit()
        
        return jsonify({'message': 'Data split successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/get-sentiment-comparison', methods=['GET'])
def get_sentiment_comparison():
    try:
        positive_count = db.session.query(TweetTraining).filter(TweetTraining.sentiment >= 0.05).count()
        negative_count = db.session.query(TweetTraining).filter(TweetTraining.sentiment <= -0.05).count()
        neutral_count = db.session.query(TweetTraining).filter((TweetTraining.sentiment > -0.05) & (TweetTraining.sentiment < 0.05)).count()
        
        sentiment_data = [
            {'id': 0, 'value': neutral_count, 'label': 'Netral'},
            {'id': 1, 'value': positive_count, 'label': 'Positif'},
            {'id': 2, 'value': negative_count, 'label': 'Negatif'}
        ]
        
        return jsonify(sentiment_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-sentiment-comparison-testing', methods=['GET'])
def get_sentiment_comparison_testing():
    try:
        positive_count = db.session.query(TweetTesting).filter(TweetTesting.sentiment >= 0.05).count()
        negative_count = db.session.query(TweetTesting).filter(TweetTesting.sentiment <= -0.05).count()
        neutral_count = db.session.query(TweetTesting).filter((TweetTesting.sentiment > -0.05) & (TweetTesting.sentiment < 0.05)).count()
        
        sentiment_data = [
            {'id': 0, 'value': neutral_count, 'label': 'Netral'},
            {'id': 1, 'value': positive_count, 'label': 'Positif'},
            {'id': 2, 'value': negative_count, 'label': 'Negatif'}
        ]
        
        return jsonify(sentiment_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get-wordcloud-data', methods=['GET'])
def get_wordcloud_data():
    try:    
        tweets = TweetTraining.query.all()
        positive_tweets = TweetTraining.query.filter(TweetTraining.sentiment >= 0.05).all()  # Positif
        negative_tweets = TweetTraining.query.filter(TweetTraining.sentiment <= -0.05).all()  # Negatif
        
        all_text = ' '.join([tweet.processed_text for tweet in tweets])
        positive_words = ' '.join([tweet.processed_text for tweet in positive_tweets])
        negative_words = ' '.join([tweet.processed_text for tweet in negative_tweets])
        
        word_counts = Counter(all_text.split())
        positive_word_counts = Counter(positive_words.split())
        negative_word_counts = Counter(negative_words.split())
        
        words_wordcloud_data = [{'text': word, 'value': count} for word, count in word_counts.items()]
        positive_wordcloud_data = [{'text': word, 'value': count} for word, count in positive_word_counts.items()]
        negative_wordcloud_data = [{'text': word, 'value': count} for word, count in negative_word_counts.items()]
        
        return jsonify({'words': words_wordcloud_data, 'positive': positive_wordcloud_data, 'negative': negative_wordcloud_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get-wordcloud-data-testing', methods=['GET'])
def get_wordcloud_data_testing():
    try:    
        tweets = TweetTesting.query.all()
        positive_tweets = TweetTesting.query.filter(TweetTesting.sentiment >= 0.05).all()  # Positif
        negative_tweets = TweetTesting.query.filter(TweetTesting.sentiment <= -0.05).all()  # Negatif
        
        all_text = ' '.join([tweet.processed_text for tweet in tweets])
        positive_words = ' '.join([tweet.processed_text for tweet in positive_tweets])
        negative_words = ' '.join([tweet.processed_text for tweet in negative_tweets])
        
        word_counts = Counter(all_text.split())
        positive_word_counts = Counter(positive_words.split())
        negative_word_counts = Counter(negative_words.split())
        
        words_wordcloud_data = [{'text': word, 'value': count} for word, count in word_counts.items()]
        positive_wordcloud_data = [{'text': word, 'value': count} for word, count in positive_word_counts.items()]
        negative_wordcloud_data = [{'text': word, 'value': count} for word, count in negative_word_counts.items()]
        
        return jsonify({'words': words_wordcloud_data, 'positive': positive_wordcloud_data, 'negative': negative_wordcloud_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def euclidean_distance(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    return distance

def calculate_tf(document):
    tf_document = {}
    total_words = len(document)
    for word in document:
        tf_document[word] = tf_document.get(word, 0) + 1 / total_words
    return tf_document

def calculate_idf(documents):
    idf = {}
    total_documents = len(documents)
    for document in documents:
        unique_words = set(document)
        for word in unique_words:
            idf[word] = idf.get(word, 0) + 1
    for word, val in idf.items():
        idf[word] = math.log(total_documents / val)
    return idf
    
def calculate_tfidf(documents):
    tfidf_documents = []
    idf = calculate_idf(documents)
    
    # Definisikan word_index
    word_index = {word: idx for idx, word in enumerate(idf.keys())}
    
    for document in documents:
        tfidf_document = np.zeros(len(idf))  # Inisialisasi vektor dengan nol
        tf_document = calculate_tf(document)
        for word, tf_val in tf_document.items():
            if word in idf:  # Pastikan kata ada dalam idf
                tfidf_document[word_index[word]] = tf_val * idf[word]  # Isi nilai TF-IDF
        tfidf_documents.append(tfidf_document)
    return tfidf_documents

def predict_sentiment_knn(all_tweets, tfidf_documents, k):
    sentiments = {}
    
    for idx, tweet in enumerate(all_tweets):
        if tweet.username.lower() in ignored_usernames:
            continue  # Skip ignored tweets
        
        # Hitung jarak ke semua tweet lain
        distances = []
        for i, other_tweet in enumerate(all_tweets):
            if i == idx or other_tweet.username.lower() in ignored_usernames:
                continue
            # Perbaiki ini
            distance = euclidean_distance(tfidf_documents[idx], tfidf_documents[i])
            distances.append((i, distance))
        
        # Temukan k tetangga terdekat
        distances.sort(key=lambda x: x[1])
        neighbors = [item[0] for item in distances[:k]]
        
        # Dapatkan sentimen tetangga
        neighbor_sentiments = [all_tweets[i].sentiment for i in neighbors]
        
        # Tentukan mayoritas sentimen dari tetangga
        majority_sentiment = Counter(neighbor_sentiments).most_common(1)[0][0]
        
        # Ubah nilai sentimen berdasarkan kriteria baru
        if majority_sentiment >= 0.05:
            majority_sentiment = 1  # Positif
        elif majority_sentiment <= -0.05:
            majority_sentiment = -1  # Negatif
        else:
            majority_sentiment = 0  # Netral
        
        sentiments[idx] = majority_sentiment
    
    # Hitung jumlah tweet yang diprediksi sebagai positif, negatif, dan netral
    positive_count = list(sentiments.values()).count(1)
    negative_count = list(sentiments.values()).count(-1)
    neutral_count = list(sentiments.values()).count(0)
    
    return positive_count, negative_count, neutral_count

# @app.route('/tf-idf', methods=['POST'])
# def calculate_tfidf():
#     try:
#         documents = [tweet.processed_text for tweet in Tweet.query.all()]
        
#         tfidf_vectorizer = TfidfVectorizer()
        
#         tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        
#         features = tfidf_vectorizer.get_feature_names_out()
        
#         tfidf_values = tfidf_matrix.toarray()
        
#         tfidf_data = [{'word': word, 'tfidf_values': tfidf_values[:, idx].tolist()} for idx, word in enumerate(features)]
        
#         return jsonify({'tfidf_data': tfidf_data}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/predict-sentiment', methods=['POST'])
def predict_sentiment_using_knn():
    try:
        all_tweets = TweetTraining.query.all()
        documents = [tweet.processed_text for tweet in all_tweets]
        
        # TF-IDF
        # tfidf_vectorizer = TfidfVectorizer()
        # tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        
        tfidf_documents = calculate_tfidf(documents)
        
        positive_count, negative_count, neutral_count = predict_sentiment_knn(all_tweets, tfidf_documents, k=9)
        
        return jsonify({'positive_count': positive_count, 'negative_count': negative_count, 'neutral_count': neutral_count}), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred during sentiment prediction using KNN: {}'.format(str(e))}), 500
    
@app.route('/predict-sentiment-testing', methods=['POST'])
def predict_sentiment_using_knn_testing():
    try:
        all_tweets = TweetTesting.query.all()
        documents = [tweet.processed_text for tweet in all_tweets]
        
        # TF-IDF
        # tfidf_vectorizer = TfidfVectorizer()
        # tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        
        tfidf_documents = calculate_tfidf(documents)
        
        positive_count, negative_count, neutral_count = predict_sentiment_knn(all_tweets, tfidf_documents, k=4)
        
        return jsonify({'positive_count': positive_count, 'negative_count': negative_count, 'neutral_count': neutral_count}), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred during sentiment prediction using KNN: {}'.format(str(e))}), 500
    
# def calculate_precision_recall_f1_score(actual_sentiments, predicted_sentiments):
#     labels = [-1, 0, 1]
#     confusion_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Inisialisasi confusion matrix
    
#     for actual, predicted in zip(actual_sentiments, predicted_sentiments):
#         if actual in labels and predicted in labels:
#             confusion_mat[labels.index(actual)][labels.index(predicted)] += 1
    
#     precision = {}
#     recall = {}
#     f1_score = {}

#     for label in range(len(labels)):
#         tp = confusion_mat[label][label]
#         fp = sum(confusion_mat[label]) - tp
#         fn = sum(confusion_mat[i][label] for i in range(len(labels))) - tp

#         precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0

#     # Menghitung nilai rata-rata dari precision, recall, dan f1_score
#     precision_avg = sum(precision.values()) / len(labels)
#     recall_avg = sum(recall.values()) / len(labels)
#     f1_score_avg = sum(f1_score.values()) / len(labels)

#     return precision_avg, recall_avg, f1_score_avg
    
@app.route('/calculate-accuracy', methods=['POST'])
def calculate_accuracy():
    try:
        all_tweets = TweetTraining.query.all()
        documents = [tweet.processed_text for tweet in all_tweets]
        
        tfidf_documents = calculate_tfidf(documents)
        
        # Prediksi sentimen menggunakan KNN
        positive_count, negative_count, neutral_count = predict_sentiment_knn(all_tweets, tfidf_documents, k=9)
        
        # Ambil sentimen aktual
        actual_sentiments = [tweet.sentiment for tweet in all_tweets]
        
        # Ubah nilai sentimen aktual berdasarkan kriteria baru
        actual_sentiments = [1 if sentiment >= 0.05 else -1 if sentiment <= -0.05 else 0 for sentiment in actual_sentiments]
        
        # Buat list sentimen prediksi berdasarkan mayoritas sentimen yang dihasilkan oleh KNN
        predicted_sentiments = [1] * positive_count + [-1] * negative_count + [0] * neutral_count
        
        # Hitung confusion matrix
        confusion_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Inisialisasi confusion matrix
        
        for actual, predicted in zip(actual_sentiments, predicted_sentiments):
            if actual == -1:
                if predicted == -1:
                    confusion_mat[0][0] += 1
                elif predicted == 0:
                    confusion_mat[0][1] += 1
                else:
                    confusion_mat[0][2] += 1
            elif actual == 0:
                if predicted == -1:
                    confusion_mat[1][0] += 1
                elif predicted == 0:
                    confusion_mat[1][1] += 1
                else:
                    confusion_mat[1][2] += 1
            else:
                if predicted == -1:
                    confusion_mat[2][0] += 1
                elif predicted == 0:
                    confusion_mat[2][1] += 1
                else:
                    confusion_mat[2][2] += 1
                    
        # Hitung metrik evaluasi
        precision, recall, f1_score, _ = precision_recall_fscore_support(actual_sentiments, predicted_sentiments, labels=[-1, 0, 1], average='weighted')
        
        precision_percent = precision * 100
        recall_percent = recall * 100
        f1_score_percent = f1_score * 100
        
        # Hitung akurasi
        correct_predictions = sum(1 for actual, predicted in zip(actual_sentiments, predicted_sentiments) if actual == predicted)
        total_predictions = len(actual_sentiments)
        accuracy = correct_predictions / total_predictions * 100
        
        return jsonify({
            'confusion_matrix': confusion_mat,
            'precision': precision_percent,
            'recall': recall_percent,
            'f1_score': f1_score_percent,
            'accuracy': accuracy
        }), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred during accuracy calculation: {}'.format(str(e))}), 500
    
@app.route('/calculate-accuracy-testing', methods=['POST'])
def calculate_accuracy_testing():
    try:
        all_tweets = TweetTesting.query.all()
        documents = [tweet.processed_text for tweet in all_tweets]
        
        tfidf_documents = calculate_tfidf(documents)
        
        # Prediksi sentimen menggunakan KNN
        positive_count, negative_count, neutral_count = predict_sentiment_knn(all_tweets, tfidf_documents, k=4)
        
        # Ambil sentimen aktual
        actual_sentiments = [tweet.sentiment for tweet in all_tweets]
        
        # Ubah nilai sentimen aktual berdasarkan kriteria baru
        actual_sentiments = [1 if sentiment >= 0.05 else -1 if sentiment <= -0.05 else 0 for sentiment in actual_sentiments]
        
        # Buat list sentimen prediksi berdasarkan mayoritas sentimen yang dihasilkan oleh KNN
        predicted_sentiments = [1] * positive_count + [-1] * negative_count + [0] * neutral_count
        
        # Hitung metrik evaluasi
        confusion_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Inisialisasi confusion matrix
        
        for actual, predicted in zip(actual_sentiments, predicted_sentiments):
            if actual == -1:
                if predicted == -1:
                    confusion_mat[0][0] += 1
                elif predicted == 0:
                    confusion_mat[0][1] += 1
                else:
                    confusion_mat[0][2] += 1
            elif actual == 0:
                if predicted == -1:
                    confusion_mat[1][0] += 1
                elif predicted == 0:
                    confusion_mat[1][1] += 1
                else:
                    confusion_mat[1][2] += 1
            else:
                if predicted == -1:
                    confusion_mat[2][0] += 1
                elif predicted == 0:
                    confusion_mat[2][1] += 1
                else:
                    confusion_mat[2][2] += 1
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(actual_sentiments, predicted_sentiments, labels=[-1, 0, 1], average='weighted')
        
        precision_percent = precision * 100
        recall_percent = recall * 100
        f1_score_percent = f1_score * 100
        
        # Hitung akurasi
        correct_predictions = sum(1 for actual, predicted in zip(actual_sentiments, predicted_sentiments) if actual == predicted)
        total_predictions = len(actual_sentiments)
        accuracy = correct_predictions / total_predictions * 100
        
        return jsonify({
            'confusion_matrix': confusion_mat,
            'precision': precision_percent,
            'recall': recall_percent,
            'f1_score': f1_score_percent,
            'accuracy': accuracy
        }), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred during accuracy calculation: {}'.format(str(e))}), 500

if __name__ == '__main__':
    app.run(debug=True)
