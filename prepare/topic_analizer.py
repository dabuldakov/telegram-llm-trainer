import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Отключаем параллелизм токенизаторов

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging

class TopicAnalyzer:
    def __init__(self, embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Инициализация с возможностью выбора модели эмбеддингов"""
        self.logger = logging.getLogger(__name__)
        self.morph = MorphAnalyzer()
        self.stop_words = get_stop_words('ru')
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.kw_model = KeyBERT(model=self.embedding_model)
        except Exception as e:
            self.logger.error(f"Ошибка загрузки моделей: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Улучшенная лемматизация и очистка текста"""
        if not text.strip():
            return ""

        words = text.split()
        cleaned_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.stop_words and word.isalpha():
                try:
                    parsed = self.morph.parse(word)[0]
                    normalized = parsed.normal_form
                    cleaned_words.append(normalized)
                except:
                    cleaned_words.append(word_lower)
        return ' '.join(cleaned_words)

    def _vectorize_texts(self, texts: List[str]) -> Tuple[Optional[TfidfVectorizer], Optional[np.ndarray]]:
        """Улучшенная векторизация текстов с обработкой ошибок"""
        if not texts or all(not text.strip() for text in texts):
            self.logger.warning("Пустые тексты для векторизации")
            return None, None

        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,  # Увеличено для лучшего покрытия
                stop_words=self.stop_words,
                ngram_range=(1, 2),  # Расширенный диапазон n-грамм
                min_df=5,           # Более строгий фильтр редких слов
                max_df=0.5,         # Более мягкий фильтр частых слов
                sublinear_tf=True   # Логарифмическое масштабирование TF
            )
            X = vectorizer.fit_transform(texts)
            return vectorizer, X
        except Exception as e:
            self.logger.error(f"Ошибка векторизации: {e}")
            return None, None

    def analyze_topics(self, 
                  sessions: List[List[Dict[str, Any]]],
                  n_topics: Optional[int],
                  min_topic_size: int,
                  batch_size: int) -> Tuple[List[int], Dict[int, List[str]]]:
        """
        Кластеризация тем с батчевой обработкой.
        """
        all_clusters = []
        all_topic_keywords = {}
        batch_start = 0
        cluster_offset = 0

        while batch_start < len(sessions):
            batch = sessions[batch_start:batch_start + batch_size]
            session_texts = [
                ' '.join(msg.get('text', '') for msg in session if msg.get('text')) 
                for session in batch
            ]
            preprocessed_texts = [self.preprocess_text(text) for text in session_texts if text.strip()]
            
            if not preprocessed_texts:
                batch_start += batch_size
                continue

            vectorizer, X = self._vectorize_texts(preprocessed_texts)
            if X is None:
                batch_start += batch_size
                continue

            # Определяем число тем для батча
            local_n_topics = n_topics
            if local_n_topics is None:
                local_n_topics = self._find_optimal_clusters(X)
                self.logger.info(f"Автоматически определено число тем в батче: {local_n_topics}")

            try:
                kmeans = KMeans(
                    n_clusters=min(local_n_topics, len(preprocessed_texts)),
                    random_state=42,
                    n_init=10
                )
                clusters = kmeans.fit_predict(X)
            except Exception as e:
                self.logger.error(f"Ошибка кластеризации: {e}")
                batch_start += batch_size
                continue

            # Сохраняем кластеры с учетом смещения
            all_clusters.extend([c + cluster_offset for c in clusters])

            # Извлекаем ключевые слова для каждого кластера в батче
            for cluster_id in range(local_n_topics):
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                if len(cluster_indices) < min_topic_size:
                    all_topic_keywords[cluster_id + cluster_offset] = ["Мало данных"]
                    continue

                cluster_texts = [preprocessed_texts[i] for i in cluster_indices]
                combined_text = ' '.join(cluster_texts)
                keywords = self._extract_combined_keywords(
                    combined_text, 
                    vectorizer,
                    kmeans.cluster_centers_[cluster_id]
                )
                all_topic_keywords[cluster_id + cluster_offset] = keywords

            cluster_offset += local_n_topics
            batch_start += batch_size

        return all_clusters, all_topic_keywords

    def _find_optimal_clusters(self, X, max_k: int = 10) -> int:
        """Определение оптимального числа кластеров методом локтя"""
        from sklearn.metrics import silhouette_score
        
        if X.shape[0] <= 2:  # Недостаточно данных
            return 1

        max_k = min(max_k, X.shape[0] - 1)
        inertias = []
        silhouette_scores = []
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                silhouette_scores.append(silhouette_score(X, labels))

        # Метод локтя по инерции
        diffs = np.diff(inertias)
        relative_diffs = diffs[:-1] / diffs[1:]
        optimal_k = np.argmax(relative_diffs) + 2 if len(relative_diffs) > 0 else 1

        # Проверка по силуэтному коэффициенту
        if silhouette_scores:
            silhouette_k = np.argmax(silhouette_scores) + 2
            optimal_k = max(optimal_k, silhouette_k)

        return min(optimal_k, max_k)

    def _extract_combined_keywords(self, 
                                 text: str, 
                                 vectorizer: TfidfVectorizer,
                                 cluster_center: np.ndarray,
                                 n_keywords: int = 5) -> List[str]:
        """Комбинированное извлечение ключевых слов"""
        # 1. Попытка KeyBERT
        try:
            if len(text.split()) >= 10:  # KeyBERT работает лучше на длинных текстах
                keywords = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words=self.stop_words,
                    top_n=n_keywords * 2,  # Берем больше для последующей фильтрации
                    use_mmr=True,
                    diversity=0.7
                )
                keywords = [kw[0] for kw in keywords if kw[1] > 0.4]  # Фильтр по confidence
                if keywords:
                    return keywords[:n_keywords]
        except Exception as e:
            self.logger.warning(f"KeyBERT не смог извлечь ключевые слова: {e}")

        # 2. Fallback на TF-IDF
        if vectorizer is not None:
            feature_names = vectorizer.get_feature_names_out()
            top_indices = cluster_center.argsort()[-n_keywords * 2:][::-1]
            tfidf_keywords = [feature_names[i] for i in top_indices 
                            if feature_names[i] in text][:n_keywords]
            return tfidf_keywords if tfidf_keywords else ["Не удалось извлечь ключевые слова"]

        return ["Не удалось извлечь ключевые слова"]