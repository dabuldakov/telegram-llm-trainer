import json
from datetime import datetime
from typing import List, Dict, Any
import re
from pathlib import Path

from datasets import Dataset
from keybert import KeyBERT
from tqdm import tqdm
from config import Config
from analize.analize import statistic
from prepare.topic_analizer import TopicAnalyzer

class ChatProcessor:
    def __init__(self):
        self.config = Config()
        self.kw_model = KeyBERT()  # Initialize once
        
    def load_and_preprocess(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and preprocess chat data from JSON file.
        
        Args:
            file_path: Path to JSON chat history file
            
        Returns:
            List of preprocessed message dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading chat data: {str(e)}")
            
        messages = []
        for msg in data.get('messages', []):
            if msg.get('type') != 'message' or not isinstance(msg.get('text'), str):
                continue
                
            # Clean text
            text = re.sub(r'\s+', ' ', msg['text']).strip()
            if not text:
                continue
                
            try:
                msg_datetime = datetime.strptime(msg['date'], '%Y-%m-%dT%H:%M:%S')
                messages.append({
                    'id': msg['id'],
                    'datetime': msg_datetime,
                    'author': msg.get('from', 'Unknown'),
                    'author_id': msg.get('from_id', ''),
                    'text': text,
                    'unix_time': int(msg['date_unixtime'])
                })
            except (KeyError, ValueError) as e:
                continue
                
        return messages
    
    def auto_segment(self, 
                   messages: List[Dict[str, Any]], 
                   time_threshold: float = 60.0, 
                   context_window: int = 5) -> List[List[Dict[str, Any]]]:
        """Segment messages into conversation sessions.
        
        Args:
            messages: List of preprocessed messages
            time_threshold: Minutes between messages to consider new session
            context_window: Number of messages to consider for context
            
        Returns:
            List of message sessions
        """
        if not messages:
            return []
            
        messages.sort(key=lambda x: x['unix_time'])
        sessions = []
        current_session = []
        
        for i in tqdm(range(len(messages)), desc="Segmenting messages"):
            if not current_session:
                current_session.append(messages[i])
                continue
                
            time_diff = (messages[i]['datetime'] - current_session[-1]['datetime']).total_seconds() / 60
            
            if (time_diff > time_threshold or 
                (len(current_session) >= context_window and 
                 not self._is_context_related(current_session, messages[i]))):
                if len(current_session) >= 2:
                    sessions.append(current_session)
                current_session = []
                
            current_session.append(messages[i])
        
        if len(current_session) >= 2:
            sessions.append(current_session)
        
        return sessions
    
    def _is_context_related(self, 
                          session: List[Dict[str, Any]], 
                          new_message: Dict[str, Any],
                          similarity_threshold: float = 0.4) -> bool:
        """Check if new message is contextually related to session."""
        session_text = ' '.join([msg['text'] for msg in session])
        new_text = new_message['text']
        
        # Extract keywords once per session/message
        keywords_session = {kw[0] for kw in 
                          self.kw_model.extract_keywords(
                              session_text, 
                              keyphrase_ngram_range=(1, 2),
                              stop_words=None)}
        
        keywords_new = {kw[0] for kw in 
                      self.kw_model.extract_keywords(
                          new_text,
                          keyphrase_ngram_range=(1, 2),
                          stop_words=None)}
        
        similarity = len(keywords_session & keywords_new) / max(1, len(keywords_session | keywords_new))
        return similarity >= similarity_threshold
    
    def format_for_llm(self, 
                      sessions: List[List[Dict[str, Any]]], 
                      clusters: List[int],
                      topic_keywords: Dict[int, List[str]]) -> str:
        """Format sessions for LLM consumption."""
        llm_data = []
        
        for i, session in enumerate(sessions):
            start_time = session[0]['datetime'].strftime('%H:%M')
            end_time = session[-1]['datetime'].strftime('%H:%M')
            day_date = session[0]['datetime'].strftime('%d-%m-%Y')
            topic_id = clusters[i]
            
            session_header = (
                f"### Сессия {i} ({start_time} - {end_time}) {day_date} "
                f"[Тема: {', '.join(topic_keywords[topic_id][:10])}]"
            )
            session_content = '\n'.join(
                [f"{msg['author']}: {msg['text']}" for msg in session])
            
            llm_data.append(f"{session_header}\n{session_content}\n")
        
        return llm_data
    
    def process_chat_data(self, topic_analizer: TopicAnalyzer) -> Dataset:
        """Main processing pipeline."""
        try:
            # 1. Load and preprocess
            messages = self.load_and_preprocess(self.config.CHAT_HISTORY_PATH)
            
            # 2. Segment
            sessions = self.auto_segment(
                messages,
                time_threshold=120,  # 2 hours
                context_window=50)
            
            # 3. Topic analysis
            clusters, topic_keywords = topic_analizer.analyze_topics(sessions, n_topics=5)
            
            # 4. Format for LLM
            llm_dataset = self.format_for_llm(sessions, clusters, topic_keywords)
            
            # 5. Save results
            with open(self.config.TEXT_DATA_FOR_LLM_SAVE_PATH, 'w', encoding='utf-8') as f:
                f.write('\n'.join(llm_dataset))

            # Create and save dataset  
            dataset = Dataset.from_dict({"text": llm_dataset})
            dataset.save_to_disk(self.config.DATA_SET_PATH)
            
            # Print analytics
            print(f"Processed messages: {len(messages)}")
            print(f"Identified sessions: {len(sessions)}")
            print("Discussion topics:")
            for topic_id, keywords in topic_keywords.items():
                print(f"Topic {topic_id}: {', '.join(keywords)}")

            # Print analize
            statistic(llm_dataset)    
            
            return dataset
            
        except Exception as e:
            print(f"Error processing chat data: {str(e)}")
            raise

processor = ChatProcessor()
topic_analizer = TopicAnalyzer()
final_dataset = processor.process_chat_data(topic_analizer)