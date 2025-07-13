from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple
import numpy as np
from stop_words import get_stop_words

class TopicAnalyzer:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.russian_stop_words = get_stop_words('ru')
        self.english_stop_words = get_stop_words('en')
        
    def preprocess_text(self, text: str) -> str:
        """Лемматизация и очистка текста на русском"""
        words = text.split()
        cleaned_words = []
        for word in words:
            if word.lower() not in self.russian_stop_words and word.lower() not in self.english_stop_words:
                parsed = self.morph.parse(word)[0]
                normalized = parsed.normal_form
                cleaned_words.append(normalized)
        return ' '.join(cleaned_words)

    def analyze_topics(self, 
                      sessions: List[List[Dict[str, Any]]], 
                      n_topics: int = 8) -> Tuple[List[int], Dict[int, List[str]]]:
        """Кластеризация сообщений по темам с улучшенной обработкой русского языка"""
        # Предварительная обработка текстов
        session_texts = [' '.join([msg['text'] for msg in session]) for session in sessions]
        preprocessed_texts = [self.preprocess_text(text) for text in session_texts]
        
        # Векторизация с учетом русской морфологии
        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words=list(self.russian_stop_words),
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9)
        
        X = vectorizer.fit_transform(preprocessed_texts)
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Извлечение ключевых слов
        topic_keywords = {}
        feature_names = vectorizer.get_feature_names_out()
        
        for cluster_id in range(n_topics):
            # Получаем центроид кластера
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Берем топ-10 слов ближайших к центру кластера
            top_indices = centroid.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            topic_keywords[cluster_id] = keywords
        
        return clusters, topic_keywords