import json
from datasets import Dataset
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from keybert import KeyBERT
import re
from config import Config

text_data_for_llm_save_path = Config.TEXT_DATA_FOR_LLM_SAVE_PATH
cgat_history_path = Config.CHAT_HISTARY_PATH
data_set_path = Config.DATA_SET_PATH

# 1. Загрузка и предобработка данных
def load_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = []
    for msg in data['messages']:
        if msg['type'] != 'message' or not isinstance(msg['text'], str):
            continue
            
        # Очистка текста
        text = re.sub(r'\s+', ' ', msg['text']).strip()
        if not text:
            continue
            
        messages.append({
            'id': msg['id'],
            'datetime': datetime.strptime(msg['date'], '%Y-%m-%dT%H:%M:%S'),
            'author': msg.get('from', 'Unknown'),
            'author_id': msg.get('from_id', ''),
            'text': text,
            'unix_time': int(msg['date_unixtime'])
        })
    
    return messages

# 2. Автоматическая сегментация
def auto_segment(messages, time_threshold=60, context_window=5):
    messages.sort(key=lambda x: x['unix_time'])
    sessions = []
    current_session = []
    
    for i in range(len(messages)):
        if not current_session:
            current_session.append(messages[i])
            continue
            
        time_diff = (messages[i]['datetime'] - current_session[-1]['datetime']).total_seconds() / 60
        
        # Проверка временного и контекстного разрыва
        if time_diff > time_threshold or (len(current_session) >= context_window and not is_context_related(current_session, messages[i])):
            if len(current_session) >= 2:  # Минимум 2 сообщения в сессии
                sessions.append(current_session)
            current_session = []
            
        current_session.append(messages[i])
    
    if len(current_session) >= 2:
        sessions.append(current_session)
    
    return sessions

def is_context_related(session, new_message, similarity_threshold=0.4):
    """Проверка семантической связанности через KeyBERT"""
    kw_model = KeyBERT()
    session_text = ' '.join([msg['text'] for msg in session])
    new_text = new_message['text']
    
    keywords_session = set([kw[0] for kw in kw_model.extract_keywords(session_text, keyphrase_ngram_range=(1, 2), stop_words=None)])
    keywords_new = set([kw[0] for kw in kw_model.extract_keywords(new_text, keyphrase_ngram_range=(1, 2), stop_words=None)])
    
    # Простая мера схожести
    similarity = len(keywords_session & keywords_new) / max(1, len(keywords_session | keywords_new))
    return similarity >= similarity_threshold

# 3. Тематический анализ
def analyze_topics(sessions, n_topics=8):
    kw_model = KeyBERT()
    session_texts = [' '.join([msg['text'] for msg in session]) for session in sessions]
    
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=['russian', 'english'])
    X = vectorizer.fit_transform(session_texts)
    
    # Кластеризация
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Извлечение ключевых слов
    topic_keywords = {}
    for cluster_id in range(n_topics):
        cluster_texts = [session_texts[i] for i in range(len(sessions)) if clusters[i] == cluster_id]
        combined_text = ' '.join(cluster_texts)
        keywords = kw_model.extract_keywords(combined_text, 
                                           keyphrase_ngram_range=(1, 2), 
                                           stop_words=None,
                                           top_n=5)
        topic_keywords[cluster_id] = [kw[0] for kw in keywords]
    
    return clusters, topic_keywords

# 4. Форматирование для LLM
def format_for_llm(sessions, clusters, topic_keywords):
    llm_data = []
    
    for i, session in enumerate(sessions):
        start_time = session[0]['datetime'].strftime('%H:%M')
        end_time = session[-1]['datetime'].strftime('%H:%M')
        topic_id = clusters[i]
        
        session_header = f"### Сессия {i} ({start_time} - {end_time}) [Тема: {', '.join(topic_keywords[topic_id][:3])}]"
        session_content = '\n'.join([f"{msg['author']}: {msg['text']}" for msg in session])
        
        llm_data.append(f"{session_header}\n{session_content}\n")
    
    return '\n'.join(llm_data)

# Основной пайплайн
def process_chat_data(input_file, output_file):
    # 1. Загрузка
    messages = load_and_preprocess(input_file)
    
    # 2. Сегментация
    sessions = auto_segment(messages, 
                          time_threshold=120,  # 2 часа между сессиями
                          context_window=7)     # 7 сообщений для контекста
    
    # 3. Тематический анализ
    clusters, topic_keywords = analyze_topics(sessions, n_topics=10)
    
    # 4. Форматирование
    llm_dataset = format_for_llm(sessions, clusters, topic_keywords)
    
    # 5. Сохранение
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(llm_dataset)
    
    # Дополнительная аналитика
    print(f"Обработано сообщений: {len(messages)}")
    print(f"Выделено сессий: {len(sessions)}")
    print(f"Темы обсуждений:")
    for topic_id, keywords in topic_keywords.items():
        print(f"Тема {topic_id}: {', '.join(keywords)}")
    
    return llm_dataset

# Запуск обработки
final_dataset = process_chat_data(cgat_history_path, text_data_for_llm_save_path)

# Создаем и сохраняем датасет
dataset = Dataset.from_dict({"text": final_dataset})
dataset.save_to_disk(data_set_path)