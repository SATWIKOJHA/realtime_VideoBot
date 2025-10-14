# cache.py
import json
import numpy as np
import re
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger("Cache")

# Lazy-load embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        MODEL_PATH = Path("models/all-MiniLM-L6-v2")
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH.absolute()}")
            raise FileNotFoundError("Download the model first!")
        logger.info("🧠 Loading sentence transformer model...")
        _embedding_model = SentenceTransformer(str(MODEL_PATH))
        logger.info("✅ Model loaded successfully!")
    return _embedding_model

CACHE_FILE = Path("cache") / "semantic_cache.json"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
LOG_FILE = Path("video_bot_wav2lip.log")

class CacheManager:
    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self._embeddings = []
        self._entries = []
        self._load_or_recover_cache()

    def _load_or_recover_cache(self):
        """Try to load cache; if missing, recover from logs"""
        if CACHE_FILE.exists():
            self._load_cache()
        else:
            logger.info("📁 No cache found. Recovering from logs...")
            self._recover_from_logs()
            self._save_cache()

    def _load_cache(self):
        try:
            with open(CACHE_FILE, 'r') as f:
                raw_cache = json.load(f)
                # ✅ FIX: Ensure embeddings are float32
                self._embeddings = [np.array(item["embedding"], dtype=np.float32) for item in raw_cache]
                self._entries = [
                    {
                        "response_text": item["response_text"],
                        "video_url": item["video_url"],
                        "created_at": item["created_at"]
                    }
                    for item in raw_cache
                ]
            logger.info(f"✅ Loaded {len(self._entries)} cached Q&A pairs")
        except Exception as e:
            logger.warning(f"⚠️ Cache file corrupted ({e}). Recovering from logs...")
            self._recover_from_logs()

    def _recover_from_logs(self):
        """Parse logs to rebuild Q → Video mapping"""
        if not LOG_FILE.exists():
            logger.warning("⚠️ Log file not found. Starting with empty cache.")
            return

        questions = []
        video_paths = []
        current_question = None

        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Match user question
                q_match = re.search(r"processing: '(.+?)\.\.\.'", line)
                if q_match:
                    current_question = q_match.group(1).strip()
                    continue

                # Match video output
                v_match = re.search(r"Output video: (static/videos/([a-f0-9\-]+)\.mp4)", line)
                if v_match and current_question:
                    video_path = v_match.group(1)
                    video_id = v_match.group(2)
                    questions.append(current_question)
                    video_paths.append(f"/static/videos/{video_id}.mp4")
                    current_question = None

        if questions:
            logger.info(f"🔍 Recovered {len(questions)} Q&A pairs from logs")
            model = get_embedding_model()
            # ✅ FIX: Explicitly use float32
            embeddings = model.encode(questions, convert_to_tensor=False, show_progress_bar=False)
            # Ensure float32
            embeddings = [emb.astype(np.float32) for emb in embeddings]
            
            self._embeddings = embeddings
            self._entries = [
                {
                    "response_text": "[Recovered from logs]",
                    "video_url": video_url,
                    "created_at": datetime.utcnow().isoformat() + "Z"
                }
                for video_url in video_paths
            ]
            logger.info(f"✅ Recovered cache built successfully!")
        else:
            logger.info("📭 No recoverable Q&A pairs found in logs.")

    def _save_cache(self):
        try:
            raw_cache = []
            for emb, entry in zip(self._embeddings, self._entries):
                # ✅ FIX: Convert to list (JSON serializable) - already float32
                raw_cache.append({
                    "embedding": emb.tolist(),
                    "response_text": entry["response_text"],
                    "video_url": entry["video_url"],
                    "created_at": entry["created_at"]
                })
            with open(CACHE_FILE, 'w') as f:
                json.dump(raw_cache, f, indent=2)
            logger.info(f"💾 Saved {len(self._entries)} entries to cache.")
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")

    def _normalize_text(self, text: str) -> str:
        return text.strip().lower()

    def get(self, question: str) -> Optional[Tuple[str, str]]:
        if not self._embeddings:
            logger.info("🔍 Cache is empty.")
            return None

        try:
            normalized = self._normalize_text(question)
            model = get_embedding_model()
            # ✅ FIX: Ensure query embedding is float32
            query_embedding = model.encode(normalized, convert_to_tensor=False, show_progress_bar=False)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            from sentence_transformers.util import cos_sim
            # Convert stored embeddings to numpy array with consistent dtype
            stored_embeddings = np.array(self._embeddings, dtype=np.float32)
            similarities = cos_sim(query_embedding, stored_embeddings)[0]
            best_match_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_match_idx])

            logger.info(f"🔍 Query: '{question[:40]}...' | Similarity: {best_similarity:.3f}")

            if best_similarity >= self.similarity_threshold:
                entry = self._entries[best_match_idx]
                logger.info(f"✅ CACHE HIT! Reusing video: {entry['video_url']}")
                return entry["response_text"], entry["video_url"]
            else:
                logger.info("❌ No similar question found in cache.")
                return None
        except Exception as e:
            logger.error(f"❌ Cache lookup error: {e}")
            return None

    def set(self, question: str, response_text: str, video_url: str):
        try:
            normalized = self._normalize_text(question)
            model = get_embedding_model()
            embedding = model.encode(normalized, convert_to_tensor=False, show_progress_bar=False)
            # ✅ FIX: Ensure float32
            embedding = np.array(embedding, dtype=np.float32)

            self._embeddings.append(embedding)
            self._entries.append({
                "response_text": response_text,
                "video_url": video_url,
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            self._save_cache()
            logger.info(f"🆕 Cached new question: '{question[:50]}...'")
        except Exception as e:
            logger.error(f"❌ Failed to cache entry: {e}")