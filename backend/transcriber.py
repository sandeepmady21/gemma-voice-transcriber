# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multilingual Code-Switching & Speaker-Diarized Voice-to-Text Engine.
Optimized for mid-conversation & mid-sentence switching across Telugu (te),
English (en), Hindi (hi), and 99+ global languages.

Supports both:
- English / Romanized script ("roman"): Writes Telugu (Tanglish), Hindi (Hinglish),
  and English in clean, easy-to-read English Latin alphabet while still tagging
  exact spoken languages (TE / HI / EN) and English meaning.
- Native script ("native"): Writes Telugu in తెలుగు, Hindi in हिन्दी, and English in Latin.
"""

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Official open-weights 3D-Speaker CampPlus ONNX model from sherpa-onnx releases
SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
)
SPEAKER_MODEL_PATH = os.path.join(
    MODELS_DIR, "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
)

# Global lock for LiteRT-LM HTTP server (LiteRTLMServer is single-threaded)
_LITERT_LOCK = threading.Lock()

# Default priority languages: Telugu, English, Hindi first
DEFAULT_PRIORITY_LANGUAGES = ["te", "en", "hi"]

LANGUAGE_NAMES: Dict[str, str] = {
    "te": "Telugu",
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "fi": "Finnish",
    "no": "Norwegian",
    "ca": "Catalan",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "sk": "Slovak",
    "fa": "Persian",
    "lv": "Latvian",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "et": "Estonian",
    "mk": "Macedonian",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "si": "Sinhala",
    "km": "Khmer",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "am": "Amharic",
    "lo": "Lao",
    "uz": "Uzbek",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "my": "Myanmar",
    "tl": "Tagalog",
    "as": "Assamese",
    "haw": "Hawaiian",
    "ha": "Hausa",
    "jw": "Javanese",
    "su": "Sundanese",
    "yue": "Cantonese",
    "la": "Latin",
    "mi": "Maori",
    "cy": "Welsh",
}

WHISPER_SCRIPT_PROMPTS: Dict[str, str] = {
    "te": "Namaskaram, idi Telugu, Hindi mariyu English sambhashana.",
    "hi": "Namaste, yah Hindi, Telugu aur English baatcheet hai.",
    "ta": "Vanakkam, idhu Tamil matrum English uraiyaadal.",
    "kn": "Namaskara, idu Kannada mattu English sambhashane.",
    "ml": "Namaskaram, ithu Malayalam, English sambhashanamanu.",
    "mr": "Namaskar, he Marathi aani English sambhashan aahe.",
}

SPEAKER_PALETTE = [
    "#3b82f6",  # Royal Blue
    "#10b981",  # Emerald Green
    "#f59e0b",  # Warm Amber
    "#a855f7",  # Vivid Purple
    "#ec4899",  # Rose Pink
    "#06b6d4",  # Cyan
]

# Common English words for detecting mid-sentence code-switching in Roman script
COMMON_ENGLISH_WORDS = {
    "hello", "hi", "hey", "okay", "ok", "yes", "no", "yeah", "please", "sorry",
    "thank", "thanks", "you", "how", "are", "what", "is", "your", "my", "name",
    "good", "morning", "evening", "night", "today", "tomorrow", "meeting", "time",
    "call", "work", "office", "going", "come", "coming", "let", "lets", "talk",
    "speak", "speaking", "can", "will", "should", "would", "could", "have", "has",
    "had", "this", "that", "there", "here", "where", "when", "why", "who", "actual",
    "actually", "basically", "because", "switch", "between", "languages", "testing",
    "test", "one", "two", "three", "four", "five", "brother", "bro", "sir", "madam",
}

COMMON_TELUGU_ROMAN_WORDS = {
    "nenu", "meeru", "mi", "nee", "naa", "peru", "per", "ela", "unnaru", "unnava",
    "unnanu", "em", "enti", "chestunnaru", "chestunnav", "ekkadiki", "ekkada",
    "vellali", "vastunna", "cheppu", "cheppandi", "kaki", "kaaki", "pani", "paina",
    "paine", "paene", "namaskaram", "bagunnara", "bagunnanu", "avunu", "kaadu",
    "ledu", "undi", "untundi", "eroju", "repu", "manam", "matladukundam", "chala",
    "manchidi", "sare", "ippudu", "appudu", "enduku", "entha", "inka", "kani",
}

COMMON_HINDI_ROMAN_WORDS = {
    "namaste", "aap", "tum", "main", "hum", "kaise", "kaisa", "kaisi", "hain",
    "hai", "ho", "aaj", "kal", "baat", "karenge", "karna", "kya", "kyun", "kahan",
    "kab", "kon", "kaun", "mera", "meri", "tera", "tumhara", "naam", "bhai", "yaar",
    "accha", "theek", "nahi", "haan", "abhi", "chalo", "milte", "dekho", "suno",
    "bahut", "bada", "chota", "kaam", "ghar", "paani", "khana",
}


def get_language_name(code: str) -> str:
    if not code:
        return "Unknown"
    base = code.lower().split("-")[0].strip()
    for k, v in LANGUAGE_NAMES.items():
        if v.lower() == base:
            return v
    return LANGUAGE_NAMES.get(base, code.upper())


def normalize_language_code(code_or_name: str) -> str:
    if not code_or_name:
        return "en"
    s = code_or_name.lower().strip()
    if s in LANGUAGE_NAMES:
        return s
    base = s.split("-")[0].strip()
    if base in LANGUAGE_NAMES:
        return base
    for k, v in LANGUAGE_NAMES.items():
        if v.lower() == s or v.lower() in s:
            return k
    return "en"


def float32_to_wav_bytes(audio_np: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encodes a 16kHz float32 mono numpy array into 16-bit PCM WAV bytes."""
    samples = np.clip(np.asarray(audio_np, dtype=np.float32), -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype("<i2")
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


def detect_script_languages(text: str) -> List[str]:
    """Detects distinct writing scripts inside a text string."""
    scripts = []
    has_latin = False
    has_devanagari = False
    has_telugu = False
    has_tamil = False
    has_kannada = False
    has_malayalam = False

    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            has_latin = True
        elif 0x0C00 <= cp <= 0x0C7F:
            has_telugu = True
        elif 0x0900 <= cp <= 0x097F:
            has_devanagari = True
        elif 0x0B80 <= cp <= 0x0BFF:
            has_tamil = True
        elif 0x0C80 <= cp <= 0x0CFF:
            has_kannada = True
        elif 0x0D00 <= cp <= 0x0D7F:
            has_malayalam = True

    if has_telugu:
        scripts.append("te")
    if has_devanagari:
        scripts.append("hi")
    if has_tamil:
        scripts.append("ta")
    if has_kannada:
        scripts.append("kn")
    if has_malayalam:
        scripts.append("ml")
    if has_latin and scripts:
        scripts.append("en")
    return scripts


def detect_romanized_languages_and_clauses(
    text: str, primary_langs: List[str]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Analyzes a Roman-script or mixed-script utterance to identify if it contains
    mid-sentence switching between English, Telugu (Tanglish), and Hindi (Hinglish).
    Returns (all_lang_codes, clauses).
    """
    words = text.split()
    if not words:
        return primary_langs or ["en"], []

    default_lang = primary_langs[0] if primary_langs else "en"
    word_langs: List[str] = []

    for w in words:
        clean_w = re.sub(r"[^a-zA-Z\u0900-\u097F\u0C00-\u0C7F]", "", w).lower()
        if any(0x0C00 <= ord(c) <= 0x0C7F for c in clean_w):
            word_langs.append("te")
        elif any(0x0900 <= ord(c) <= 0x097F for c in clean_w):
            word_langs.append("hi")
        elif clean_w in COMMON_TELUGU_ROMAN_WORDS:
            word_langs.append("te")
        elif clean_w in COMMON_HINDI_ROMAN_WORDS:
            word_langs.append("hi")
        elif clean_w in COMMON_ENGLISH_WORDS:
            word_langs.append("en")
        else:
            word_langs.append(default_lang)

    # Smooth isolated 1-word flips unless at start/end greeting
    if len(word_langs) >= 3:
        for i in range(1, len(word_langs) - 1):
            if word_langs[i - 1] == word_langs[i + 1] and word_langs[i] != word_langs[i - 1]:
                word_langs[i] = word_langs[i - 1]

    clauses: List[Dict[str, Any]] = []
    cur_words = [words[0]]
    cur_lang = word_langs[0]

    for w, l in zip(words[1:], word_langs[1:]):
        if l == cur_lang:
            cur_words.append(w)
        else:
            clauses.append(
                {
                    "text": " ".join(cur_words),
                    "lang": cur_lang,
                    "lang_name": get_language_name(cur_lang),
                    "lang_prob": 0.95,
                }
            )
            cur_words = [w]
            cur_lang = l

    if cur_words:
        clauses.append(
            {
                "text": " ".join(cur_words),
                "lang": cur_lang,
                "lang_name": get_language_name(cur_lang),
                "lang_prob": 0.95,
            }
        )

    all_langs: List[str] = []
    for c in clauses:
        if c["lang"] not in all_langs:
            all_langs.append(c["lang"])
    if not all_langs:
        all_langs = list(primary_langs) if primary_langs else ["en"]

    return all_langs, clauses


def _compute_acoustic_voiceprint(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fallback 60-dim L2-normalized spectral + prosodic voiceprint."""
    if len(audio) < int(sr * 0.15):
        audio = np.pad(audio, (0, int(sr * 0.15) - len(audio)))
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-5:
        audio = audio / peak

    n_fft = 512
    hop = 160
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop]
    window = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-7

    frame_energy = np.mean(mag**2, axis=1)
    active_mask = frame_energy > (np.percentile(frame_energy, 25) * 0.5)
    if np.any(active_mask):
        mag = mag[active_mask]

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band_edges = np.geomspace(80.0, 7600.0, 25)
    band_energies = []
    for b in range(24):
        idx = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        if np.any(idx):
            band_energies.append(np.log(np.mean(mag[:, idx], axis=1) + 1e-6))
        else:
            band_energies.append(np.zeros(mag.shape[0], dtype=np.float32))
    band_matrix = np.stack(band_energies, axis=1)
    n_bands = 24
    k = np.arange(16)[:, None]
    n = np.arange(n_bands)[None, :]
    dct_basis = np.cos(np.pi / n_bands * (n + 0.5) * k).astype(np.float32)
    mfcc = band_matrix @ dct_basis.T
    mfcc_timbre = mfcc[:, 1:]

    feat = np.concatenate(
        [
            np.mean(mfcc_timbre, axis=0),
            np.std(mfcc_timbre, axis=0),
            np.percentile(mfcc_timbre, 25, axis=0),
            np.percentile(mfcc_timbre, 75, axis=0),
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat = feat / norm
    return feat


class SpeakerDiarizer:
    """
    Manages neural speaker embedding extraction (sherpa-onnx CampPlus 16kHz with
    acoustic voiceprint fallback) and tracks speakers across the conversation.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._extractor = None
        self._extractor_ready = False
        self._init_attempted = False
        self.speakers: List[Dict[str, Any]] = []
        self.similarity_threshold: float = 0.52

    def _ensure_extractor(self) -> None:
        if self._init_attempted:
            return
        with self._lock:
            if self._init_attempted:
                return
            self._init_attempted = True
            try:
                import requests
                import sherpa_onnx

                if not os.path.exists(SPEAKER_MODEL_PATH):
                    print(
                        f"[Diarizer] Downloading neural speaker model to {SPEAKER_MODEL_PATH}...",
                        flush=True,
                    )
                    with requests.get(SPEAKER_MODEL_URL, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(SPEAKER_MODEL_PATH + ".tmp", "wb") as f:
                            for chunk in r.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                    os.replace(SPEAKER_MODEL_PATH + ".tmp", SPEAKER_MODEL_PATH)
                    print("[Diarizer] Neural speaker model downloaded.", flush=True)

                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=SPEAKER_MODEL_PATH,
                    num_threads=2,
                    debug=False,
                    provider="cpu",
                )
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
                self._extractor_ready = True
                print("[Diarizer] sherpa-onnx neural speaker embedding extractor ready.", flush=True)
            except Exception as e:
                print(
                    f"[Diarizer] Falling back to acoustic voiceprint extractor ({e})",
                    flush=True,
                )
                self._extractor_ready = False

    def extract_embedding(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Extracts an L2-normalized speaker embedding from 16kHz float32 speech audio."""
        self._ensure_extractor()
        audio_f32 = np.asarray(audio, dtype=np.float32)

        peak = np.max(np.abs(audio_f32)) if len(audio_f32) > 0 else 0.0
        if peak > 1e-4:
            audio_f32 = (audio_f32 / peak) * 0.85

        # Tile short clips (<1.6s) smoothly so CampPlus temporal pooling is stable
        min_samples = int(sr * 1.6)
        if 0 < len(audio_f32) < min_samples:
            reps = int(np.ceil(min_samples / len(audio_f32)))
            audio_f32 = np.tile(audio_f32, reps)[:min_samples]
        elif len(audio_f32) == 0:
            audio_f32 = np.zeros(min_samples, dtype=np.float32)

        if self._extractor_ready and self._extractor is not None:
            try:
                stream = self._extractor.create_stream()
                stream.accept_waveform(sample_rate=sr, waveform=audio_f32.tolist())
                stream.input_finished()
                if self._extractor.is_ready(stream):
                    emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        return emb / norm
            except Exception as e:
                print(f"[Diarizer] ONNX embedding warning: {e}", flush=True)

        return _compute_acoustic_voiceprint(audio_f32, sr)

    def match_or_register_speaker(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Matches the audio clip against known speakers in the session, or creates
        a new speaker profile if the voice is genuinely distinct and long enough.
        """
        emb = self.extract_embedding(audio, sr)
        duration = len(audio) / float(sr)
        sim_thresh = threshold if threshold is not None else self.similarity_threshold
        if not self._extractor_ready:
            sim_thresh = max(sim_thresh, 0.78)

        with self._lock:
            if not self.speakers:
                spk = self._create_speaker(emb, duration=duration)
                return spk, 0.98

            best_spk = None
            best_sim = -1.0
            for spk in self.speakers:
                centroid = spk["centroid"]
                if centroid.shape != emb.shape:
                    continue
                sim = float(np.dot(emb, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_spk = spk

            effective_thresh = sim_thresh - (0.12 if duration < 1.25 else 0.0)

            if best_spk is not None and best_sim >= effective_thresh:
                if duration >= 1.0:
                    alpha = min(0.25, max(0.06, duration / 14.0))
                    new_centroid = (1.0 - alpha) * best_spk["centroid"] + alpha * emb
                    norm = np.linalg.norm(new_centroid)
                    if norm > 1e-6:
                        best_spk["centroid"] = new_centroid / norm
                best_spk["utterance_count"] += 1
                best_spk["total_seconds"] = round(best_spk["total_seconds"] + duration, 1)
                conf = round(max(0.65, min(0.99, best_sim + 0.25)), 2)
                return best_spk, conf

            # Only spawn a new speaker if the speech segment is at least 1.25s long
            if duration >= 1.25 and len(self.speakers) < 8:
                spk = self._create_speaker(emb, duration=duration)
                return spk, 0.92

            if best_spk is not None:
                best_spk["utterance_count"] += 1
                best_spk["total_seconds"] = round(best_spk["total_seconds"] + duration, 1)
                return best_spk, round(max(0.60, best_sim + 0.20), 2)

            spk = self._create_speaker(emb, duration=duration)
            return spk, 0.90

    def _create_speaker(
        self, emb: np.ndarray, label: Optional[str] = None, duration: float = 1.0
    ) -> Dict[str, Any]:
        idx = len(self.speakers) + 1
        spk = {
            "id": f"spk_{idx}",
            "index": idx,
            "label": label or f"Speaker {idx}",
            "color": SPEAKER_PALETTE[(idx - 1) % len(SPEAKER_PALETTE)],
            "centroid": emb.copy(),
            "utterance_count": 1,
            "total_seconds": round(duration, 1),
            "enrolled": bool(label),
        }
        self.speakers.append(spk)
        return spk

    def rename_speaker(self, speaker_id: str, new_label: str) -> bool:
        with self._lock:
            for spk in self.speakers:
                if spk["id"] == speaker_id:
                    spk["label"] = new_label.strip() or spk["label"]
                    spk["enrolled"] = True
                    return True
        return False

    def enroll_speaker(
        self, label: str, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> Dict[str, Any]:
        emb = self.extract_embedding(audio, sr)
        duration = len(audio) / float(sr)
        with self._lock:
            for spk in self.speakers:
                if spk["label"].lower() == label.strip().lower():
                    new_centroid = 0.4 * spk["centroid"] + 0.6 * emb
                    norm = np.linalg.norm(new_centroid)
                    if norm > 1e-6:
                        spk["centroid"] = new_centroid / norm
                    spk["enrolled"] = True
                    return self._serialize_speaker(spk)
            spk = self._create_speaker(emb, label=label.strip(), duration=duration)
            return self._serialize_speaker(spk)

    def reset_speakers(self) -> None:
        with self._lock:
            self.speakers.clear()

    def get_speakers_summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._serialize_speaker(s) for s in self.speakers]

    @staticmethod
    def _serialize_speaker(spk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": spk["id"],
            "index": spk["index"],
            "label": spk["label"],
            "color": spk["color"],
            "utterance_count": spk["utterance_count"],
            "total_seconds": spk["total_seconds"],
            "enrolled": spk.get("enrolled", False),
        }


from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter

GEMINI_KEY_FILE = os.path.join(MODELS_DIR, ".gemini_api_key")

# Persistent HTTP/TLS keep-alive session to eliminate 350ms SSL handshake per chunk
_HTTP_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=1)
_HTTP_SESSION.mount("https://", _adapter)
_HTTP_SESSION.mount("http://", _adapter)


def get_saved_gemini_key() -> str:
    """Returns the Gemini API key from environment or persisted local key file."""
    env_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if env_key:
        return env_key
    if os.path.exists(GEMINI_KEY_FILE):
        try:
            with open(GEMINI_KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def save_gemini_key(api_key: str) -> None:
    """Persists the Gemini API key locally in backend/models/.gemini_api_key."""
    clean = (api_key or "").strip()
    os.environ["GEMINI_API_KEY"] = clean
    try:
        with open(GEMINI_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(clean)
    except Exception as e:
        print(f"[Gemini] Failed to save key file: {e}", flush=True)


RECORDINGS_DIR = os.path.join( os.path.dirname(MODELS_DIR), "recordings" )
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Cache any models that returned 404 for this API key so we never waste latency retrying them
_DEPRECATED_MODELS_CACHE: set = set()


def _resolve_gemini_candidates(requested_model: str) -> List[str]:
    """
    Returns ordered candidate Gemini models supporting low-latency audio transcription.
    Automatically skips models deprecated for newer Google AI Studio keys (e.g. 2.5/2.0-flash 404s).
    """
    default_chain = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]
    req = (requested_model or "").strip()
    # Map legacy default names directly to the active 3.5/3.1 flash-lite chain
    if not req or req in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.1-flash-lite"):
        chain = default_chain
    else:
        chain = [req] + [m for m in default_chain if m != req]
    return [m for m in chain if m not in _DEPRECATED_MODELS_CACHE] or default_chain


def _build_gemini_gen_config(model_name: str, max_tokens: int = 180) -> Dict[str, Any]:
    gen_config: Dict[str, Any] = {
        "temperature": 0.0,
        "maxOutputTokens": max_tokens,
    }
    if any(k in model_name for k in ("3.1", "3.5", "flash-lite")):
        gen_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
    elif "2.5" in model_name:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
    return gen_config


def test_gemini_api_key(
    api_key: str, model_name: str = "gemini-3.1-flash-lite"
) -> Dict[str, Any]:
    """Tests whether a Gemini API key is valid and warms up the persistent TLS connection."""
    key = (api_key or "").strip() or get_saved_gemini_key()
    if not key:
        return {"ok": False, "error": "Missing Gemini API Key"}

    last_err = "Unknown error"
    for target_model in _resolve_gemini_candidates(model_name):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": "ok"}]}],
            "generationConfig": _build_gemini_gen_config(target_model, max_tokens=4),
        }
        try:
            resp = _HTTP_SESSION.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if "candidates" in data:
                    return {"ok": True, "model": target_model}
            try:
                err_body = resp.json()
                msg = err_body.get("error", {}).get("message", resp.text)
            except Exception:
                msg = resp.text
            last_err = msg
            if resp.status_code == 404:
                _DEPRECATED_MODELS_CACHE.add(target_model)
                continue
            if resp.status_code in (400, 401, 403) and "API key" in msg:
                return {"ok": False, "error": msg}
        except Exception as e:
            last_err = str(e)
    return {"ok": False, "error": last_err}


def extract_speech_regions_with_vad(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_segment_sec: float = 5.5,
    min_silence_ms: int = 350,
) -> List[Tuple[int, int]]:
    """
    Uses Silero Neural VAD (via faster_whisper.vad) to isolate actual human speech
    segments and discard background room noise / mic clicks.
    """
    if len(audio) < int(sr * 0.25):
        return []

    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        ts = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=0.40,
                min_speech_duration_ms=200,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=260,
            ),
            sampling_rate=sr,
        )
        if not ts:
            return []

        regions: List[Tuple[int, int]] = []
        max_samps = int(max_segment_sec * sr)
        for seg in ts:
            s_samp = max(0, int(seg["start"]))
            e_samp = min(len(audio), int(seg["end"]))
            if e_samp - s_samp < int(0.20 * sr):
                continue
            cur = s_samp
            while e_samp - cur > max_samps:
                regions.append((cur, cur + max_samps))
                cur += max_samps
            if e_samp - cur >= int(0.20 * sr):
                regions.append((cur, e_samp))
        return regions
    except Exception as e:
        print(f"[VAD] Fallback to energy VAD: {e}", flush=True)
        peak = np.max(np.abs(audio))
        if peak < 0.02:
            return []
        return [(0, len(audio))]


def transcribe_segment_with_gemini(
    sub_audio: np.ndarray,
    api_key: str,
    model_name: str = "gemini-3.5-flash-lite",
    allowed_languages: Optional[List[str]] = None,
    script_mode: str = "roman",
) -> Optional[Dict[str, Any]]:
    """
    Ultra-low-latency Gemini Cloud Audio transcription.
    Normalizes audio gain so quiet/secondary speakers are clear, uses a Telugu/English/Hindi
    conversational context prompt, and parses single or multi-block responses reliably.
    """
    key = (api_key or "").strip() or get_saved_gemini_key()
    if not key:
        raise ValueError(
            "Gemini API Key required. Click 'Add Key' in the sidebar to paste your Google AI Studio key."
        )

    # Normalize audio gain so quiet voices or speakers further from the mic are clear
    norm_audio = np.asarray(sub_audio, dtype=np.float32)
    peak = float(np.max(np.abs(norm_audio))) if len(norm_audio) > 0 else 0.0
    if peak > 0.01:
        norm_audio = np.clip((norm_audio / peak) * 0.85, -0.98, 0.98)

    wav_bytes = float32_to_wav_bytes(norm_audio, SAMPLE_RATE)
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    if allowed_languages and len(allowed_languages) > 0:
        lang_hint = ", ".join(
            f"{get_language_name(c)} ({c})" for c in allowed_languages
        )
    else:
        lang_hint = "Telugu (te), English (en), Hindi (hi)"

    if script_mode == "roman":
        prompt = (
            f"Transcribe this audio accurately ({lang_hint} / mid-sentence code-switching).\n"
            "IMPORTANT CONTEXT: The speakers are Indian speakers primarily speaking conversational Telugu (Tanglish), Indian English, or Hindi.\n"
            "Listen carefully for Telugu words (e.g., babji, mee, nenu, eroju, ekkuva, matladathadu, avunu ra, vachindi, vellanu, chala, bagundi, enti, cheppu) — do NOT confuse spoken Telugu with Hindi!\n"
            "Write Telugu in Romanized Telugu (Tanglish), Hindi in Romanized Hindi (Hinglish), and English in English.\n"
            "Do NOT translate spoken Telugu/Hindi words in TEXT. If silence/noise only, output LANG: NONE.\n"
            "Put ALL spoken words onto a SINGLE TEXT line and a SINGLE MEANING line:\n"
            "LANG: <comma-separated codes: te, en, hi, or NONE>\n"
            "TEXT: <complete verbatim spoken words in English/Roman alphabet>\n"
            "MEANING: <complete English translation>"
        )
    else:
        prompt = (
            f"Transcribe this audio accurately ({lang_hint} / mid-sentence code-switching).\n"
            "IMPORTANT CONTEXT: The speakers are Indian speakers primarily speaking conversational Telugu, Indian English, or Hindi.\n"
            "Listen carefully for Telugu speech — do NOT confuse spoken Telugu with Hindi!\n"
            "Write Telugu in తెలుగు script, Hindi in हिन्दी script, and English in Latin script.\n"
            "Do NOT translate spoken words in TEXT. If silence/noise only, output LANG: NONE.\n"
            "Put ALL spoken words onto a SINGLE TEXT line and a SINGLE MEANING line:\n"
            "LANG: <comma-separated codes: te, en, hi, or NONE>\n"
            "TEXT: <complete verbatim spoken words in native script>\n"
            "MEANING: <complete English translation>"
        )

    models_to_try = _resolve_gemini_candidates(model_name)

    last_error = None
    for candidate_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent?key={key}"
        gen_config = _build_gemini_gen_config(candidate_model, max_tokens=220)

        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": gen_config,
        }

        try:
            resp = _HTTP_SESSION.post(url, json=payload, timeout=12)
            if resp.status_code != 200:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    err_msg = resp.text
                last_error = f"Gemini API Error ({resp.status_code}): {err_msg}"
                print(f"[Gemini-Cloud] {candidate_model} error: {last_error}", flush=True)
                if resp.status_code == 404:
                    _DEPRECATED_MODELS_CACHE.add(candidate_model)
                    continue
                if resp.status_code in (400, 401, 403) and "API key" in err_msg:
                    raise ValueError(last_error)
                continue

            resp_data = resp.json()
            candidates = resp_data.get("candidates") or []
            if not candidates:
                return {"text": "", "silence": True}
            parts = candidates[0].get("content", {}).get("parts") or []
            raw_reply = "".join(p.get("text", "") for p in parts).strip()
            if not raw_reply:
                return {"text": "", "silence": True}

            lang_parts: List[str] = []
            text_parts: List[str] = []
            meaning_parts: List[str] = []

            for line in raw_reply.splitlines():
                s = line.strip().strip("*_ ")
                if not s:
                    continue
                u = s.upper()
                if u.startswith("LANG:") or u.startswith("LANGUAGE:"):
                    val = s.split(":", 1)[1].strip().strip("*_ ")
                    if val:
                        lang_parts.append(val)
                elif u.startswith("TEXT:") or u.startswith("ROMAN:") or u.startswith("TRANSCRIPT:"):
                    val = s.split(":", 1)[1].strip().strip("*_ \"'`")
                    if val and val.upper() not in ("NONE", "N/A", "SILENCE", "[SILENCE]", "<NOISE>"):
                        text_parts.append(val)
                elif u.startswith("MEANING:") or u.startswith("TRANSLATION:"):
                    val = s.split(":", 1)[1].strip().strip("*_ \"'`")
                    if val and val.upper() not in ("NONE", "N/A", "SILENCE", "[SILENCE]"):
                        meaning_parts.append(val)

            lang_line = ", ".join(lang_parts)
            text_line = " ".join(text_parts).strip()
            meaning_line = " ".join(meaning_parts).strip()

            if not text_line:
                cleaned = [
                    l.strip()
                    for l in raw_reply.splitlines()
                    if not l.strip().upper().startswith(("LANG:", "LANGUAGE:", "MEANING:", "TRANSLATION:", "NATIVE:"))
                ]
                text_line = " ".join(cleaned).strip()

            text_line = text_line.strip(" \"'`")
            meaning_line = meaning_line.strip(" \"'`")

            if (
                not text_line
                or ("NONE" in lang_line.upper() and len(text_parts) == 0)
                or text_line.upper() in ("NONE", "N/A", "SILENCE", "[SILENCE]", "NULL", "<NOISE>")
                or len(text_line) < 2
            ):
                return {"text": "", "silence": True}

            declared_langs: List[str] = []
            if lang_line:
                for part in re.split(r"[,+/&]|\band\b", lang_line, flags=re.IGNORECASE):
                    part = part.strip()
                    if part and part.upper() != "NONE":
                        code = normalize_language_code(part)
                        if code not in declared_langs:
                            declared_langs.append(code)
            if not declared_langs:
                declared_langs = ["en"]

            all_langs, clauses = detect_romanized_languages_and_clauses(
                text_line, declared_langs
            )

            return {
                "text": text_line,
                "languages": all_langs,
                "clauses": clauses,
                "english_translation": meaning_line,
            }
        except ValueError:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"[Gemini-Cloud] {candidate_model} exception: {last_error}", flush=True)
            continue

    if last_error and ("API key" in last_error or "quota" in last_error.lower()):
        raise ValueError(last_error)
    return None




def transcribe_segment_with_gemma4(
    sub_audio: np.ndarray,
    litert_url: str = "http://localhost:9379/v1/chat/completions",
    model_name: str = "gemma4-e2b",
    allowed_languages: Optional[List[str]] = None,
    script_mode: str = "roman",
) -> Optional[Dict[str, Any]]:
    """
    Streams transcription from Gemma 4 E2B's multimodal audio encoder via LiteRT-LM.
    Uses LANGUAGE-first structured output so language detection and script rendering
    are fast (~1.5-2.5s) and accurate across Telugu, Hindi, and English.
    """
    wav_bytes = float32_to_wav_bytes(sub_audio, SAMPLE_RATE)
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    active_langs = allowed_languages if allowed_languages else DEFAULT_PRIORITY_LANGUAGES
    lang_names_str = ", ".join(get_language_name(c) for c in active_langs)

    if script_mode == "native":
        prompt_text = (
            "Listen to the audio and respond in this exact 3-line format:\n"
            f"LANGUAGE: <spoken language(s): {lang_names_str}>\n"
            "TRANSCRIPT: <exact verbatim transcript in the native script of each spoken language: "
            "Telugu in Telugu script, Hindi in Devanagari script, English in Latin script>\n"
            "MEANING: <English translation>"
        )
    else:
        prompt_text = (
            "Listen to the audio and respond in this exact 3-line format:\n"
            f"LANGUAGE: <spoken language(s): {lang_names_str}>\n"
            "ROMAN: <verbatim transcript written in English Latin alphabet (Romanized Telugu/Hindi or English)>\n"
            "MEANING: <English translation>"
        )

    payload = {
        "model": model_name,
        "temperature": 0.0,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": wav_b64,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
    }

    req = urllib.request.Request(
        litert_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: List[str] = []
    try:
        with _LITERT_LOCK:
            with urllib.request.urlopen(req, timeout=22) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        evt = json.loads(data_str)
                    except Exception:
                        continue
                    delta = (
                        evt.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        chunks.append(delta)
                        joined = "".join(chunks)
                        # Stop early if MEANING line is complete and followed by newline, or if too long
                        if len(joined) > 380:
                            break
                        if "MEANING:" in joined:
                            after_meaning = joined.split("MEANING:", 1)[1]
                            if "\n" in after_meaning.lstrip():
                                break
    except Exception as e:
        print(f"[Gemma4-Audio] Inference error: {e}", flush=True)
        return None

    raw_reply = "".join(chunks).strip()
    if not raw_reply or "[SILENCE]" in raw_reply.upper():
        return None

    lang_line = ""
    text_line = ""
    meaning_line = ""

    for line in raw_reply.splitlines():
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("LANGUAGE:"):
            lang_line = s.split(":", 1)[1].strip()
        elif u.startswith("ROMAN:") or u.startswith("TRANSCRIPT:") or u.startswith("TEXT:"):
            text_line = s.split(":", 1)[1].strip()
        elif u.startswith("MEANING:") or u.startswith("TRANSLATION:"):
            meaning_line = s.split(":", 1)[1].strip()

    if not text_line:
        # Fallback if model returned plain text
        cleaned_lines = [
            l.strip()
            for l in raw_reply.splitlines()
            if not l.strip().upper().startswith(("LANGUAGE:", "MEANING:"))
        ]
        text_line = " ".join(cleaned_lines).strip()

    # Strip any accidental prompt echo
    for echo_marker in ("Listen to the audio", "respond in this exact", "3-line format"):
        if echo_marker.lower() in text_line.lower():
            idx = text_line.lower().find(echo_marker.lower())
            text_line = text_line[:idx].strip()
        if echo_marker.lower() in meaning_line.lower():
            idx = meaning_line.lower().find(echo_marker.lower())
            meaning_line = meaning_line[:idx].strip()

    # Strip stray Arabic/Cyrillic subword artifacts if script_mode == 'native'
    if script_mode == "native":
        text_line = re.sub(r"[\u0600-\u06FF]+", "", text_line).strip()

    if not text_line:
        return None

    detected_langs: List[str] = []
    if lang_line:
        for part in re.split(r"[,/|+&]+", lang_line):
            p = part.strip()
            if p:
                code = normalize_language_code(p)
                if code == "ur" and ("hi" in active_langs):
                    code = "hi"
                if code not in detected_langs:
                    detected_langs.append(code)

    if not detected_langs:
        detected_langs = detect_script_languages(text_line) or ["en"]

    all_langs, clauses = detect_romanized_languages_and_clauses(text_line, detected_langs)

    return {
        "text": text_line,
        "languages": all_langs,
        "clauses": clauses,
        "english_translation": meaning_line,
    }


class MultilingualTranscriber:
    """
    Coordinates Silero Neural VAD, sherpa-onnx 3D-Speaker diarization,
    Gemma 4 E2B multimodal audio code-switching transcription, and Faster-Whisper.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._whisper_model = None
        self._whisper_model_size = "small"
        self.diarizer = SpeakerDiarizer()

    def get_whisper_model(self, model_size: str = "small"):
        with self._lock:
            if self._whisper_model is not None and self._whisper_model_size == model_size:
                return self._whisper_model
            from faster_whisper import WhisperModel

            print(
                f"[ASR] Loading Faster-Whisper multilingual model '{model_size}' (int8 CPU)...",
                flush=True,
            )
            self._whisper_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=6,
            )
            self._whisper_model_size = model_size
            print(f"[ASR] Faster-Whisper '{model_size}' ready.", flush=True)
            return self._whisper_model

    def prewarm_litert(
        self,
        litert_url: str = "http://localhost:9379/v1/chat/completions",
        model_name: str = "gemma4-e2b",
    ) -> None:
        """Warms up LiteRT-LM Gemma 4 engine on startup so first user recording is fast."""
        try:
            payload = {
                "model": model_name,
                "temperature": 0.0,
                "stream": True,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
            }
            req = urllib.request.Request(
                litert_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _LITERT_LOCK:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    for _ in resp:
                        break
            print(f"[Prewarm] LiteRT-LM '{model_name}' engine warmed up and ready.", flush=True)
        except Exception as e:
            print(f"[Prewarm] LiteRT-LM warmup note: {e}", flush=True)

    def transcribe_conversation_chunk(
        self,
        audio_np: np.ndarray,
        engine_mode: str = "gemini-cloud",
        whisper_model_size: str = "small",
        allowed_languages: Optional[List[str]] = None,
        speaker_threshold: Optional[float] = None,
        litert_url: str = "http://localhost:9379/v1/chat/completions",
        gemma_model: str = "gemma4-e2b",
        script_mode: str = "roman",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
    ) -> Dict[str, Any]:
        """
        Processes a recorded or streamed audio buffer (16kHz mono float32):
        1. Filters out silence/noise and splits speech at natural pauses via Silero Neural VAD.
        2. Identifies the speaker for each speech phrase via sherpa-onnx 3D-Speaker voiceprints.
        3. Transcribes each phrase with Gemini 2.5 Flash Cloud (or Gemma 4 / Whisper offline)
           and groups contiguous phrases from the same speaker into rich diarized turns.
        """
        t0 = time.time()
        audio_np = np.asarray(audio_np, dtype=np.float32)
        total_duration = round(len(audio_np) / float(SAMPLE_RATE), 2)
        if len(audio_np) < int(SAMPLE_RATE * 0.25):
            return {
                "turns": [],
                "speakers": self.diarizer.get_speakers_summary(),
                "duration": total_duration,
                "processing_time": round(time.time() - t0, 2),
                "engine": engine_mode,
            }

        raw_peak = float(np.max(np.abs(audio_np)))
        raw_rms = float(np.sqrt(np.mean(audio_np * audio_np)))
        if raw_peak < 0.003 or raw_rms < 0.0008:
            return {
                "turns": [],
                "speakers": self.diarizer.get_speakers_summary(),
                "duration": total_duration,
                "processing_time": round(time.time() - t0, 2),
                "engine": engine_mode,
            }

        # Normalize audio before VAD so soft voices or external speaker/phone audio are detected reliably
        if raw_peak < 0.55:
            gain = min(18.0, 0.80 / max(raw_peak, 1e-4))
            audio_np = np.clip(audio_np * gain, -0.98, 0.98).astype(np.float32)

        is_gemini = engine_mode == "gemini-cloud"
        regions = extract_speech_regions_with_vad(
            audio_np,
            sr=SAMPLE_RATE,
            max_segment_sec=12.0 if is_gemini else 6.0,
            min_silence_ms=950 if is_gemini else 450,
        )
        # If Silero VAD missed soft/external audio on a short live chunk that clearly has signal,
        # pass the whole chunk to Gemini so live subtitles never drop out.
        if not regions:
            if total_duration <= 10.0 and raw_rms >= 0.0025:
                regions = [(0, len(audio_np))]
            else:
                return {
                    "turns": [],
                    "speakers": self.diarizer.get_speakers_summary(),
                    "duration": total_duration,
                    "processing_time": round(time.time() - t0, 2),
                    "engine": engine_mode,
                }

        valid_segments: List[Tuple[int, int, np.ndarray]] = []
        for s_samp, e_samp in regions:
            sub_audio = audio_np[s_samp:e_samp]
            if np.max(np.abs(sub_audio)) >= 0.01:
                valid_segments.append((s_samp, e_samp, sub_audio))

        if not valid_segments:
            return {
                "turns": [],
                "speakers": self.diarizer.get_speakers_summary(),
                "duration": total_duration,
                "processing_time": round(time.time() - t0, 2),
                "engine": engine_mode,
            }

        def _process_one_segment(seg_tuple: Tuple[int, int, np.ndarray]):
            s_samp, e_samp, sub_audio = seg_tuple
            # Extract speaker embedding concurrently with ASR when using Gemini Cloud
            res: Optional[Dict[str, Any]] = None
            if is_gemini:
                with ThreadPoolExecutor(max_workers=2) as inner_pool:
                    emb_future = inner_pool.submit(
                        self.diarizer.extract_embedding, sub_audio, SAMPLE_RATE
                    )
                    asr_future = inner_pool.submit(
                        transcribe_segment_with_gemini,
                        sub_audio,
                        gemini_api_key,
                        gemini_model,
                        allowed_languages,
                        script_mode,
                    )
                    res = asr_future.result()
                    _ = emb_future.result()
            elif engine_mode in ("gemma4-audio", "hybrid"):
                res = transcribe_segment_with_gemma4(
                    sub_audio,
                    litert_url=litert_url,
                    model_name=gemma_model,
                    allowed_languages=allowed_languages,
                    script_mode=script_mode,
                )

            if (not res or not res.get("text")) and not (res and res.get("silence")):
                wm = self.get_whisper_model(whisper_model_size)
                segs, info = wm.transcribe(
                    sub_audio,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                w_text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
                if w_text:
                    det_lang = normalize_language_code(info.language or "en")
                    all_langs, clauses = detect_romanized_languages_and_clauses(
                        w_text, [det_lang]
                    )
                    res = {
                        "text": w_text,
                        "languages": all_langs,
                        "clauses": clauses,
                        "english_translation": "",
                    }
            return (s_samp, e_samp, sub_audio, res)

        # Run segments in parallel when using Gemini Cloud
        if is_gemini and len(valid_segments) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(valid_segments))) as pool:
                segment_results = list(pool.map(_process_one_segment, valid_segments))
        else:
            segment_results = [_process_one_segment(seg) for seg in valid_segments]

        turns: List[Dict[str, Any]] = []
        turn_audio_buffers: List[List[np.ndarray]] = []

        for s_samp, e_samp, sub_audio, res in segment_results:
            if not res or not res.get("text"):
                continue

            text = res["text"].strip()
            if not text:
                continue

            spk_info, spk_conf = self.diarizer.match_or_register_speaker(
                sub_audio, sr=SAMPLE_RATE, threshold=speaker_threshold
            )

            lang_codes = res.get("languages") or ["en"]
            clauses = res.get("clauses") or [
                {
                    "text": text,
                    "lang": lang_codes[0],
                    "lang_name": get_language_name(lang_codes[0]),
                    "lang_prob": 0.98,
                }
            ]
            for c in clauses:
                c["start"] = round(s_samp / float(SAMPLE_RATE), 2)
                c["end"] = round(e_samp / float(SAMPLE_RATE), 2)

            eng_meaning = res.get("english_translation", "").strip()

            if turns and turns[-1]["speaker_id"] == spk_info["id"]:
                prev = turns[-1]
                prev["end"] = round(e_samp / float(SAMPLE_RATE), 2)
                prev["text"] = f"{prev['text']} {text}".strip()
                prev["clauses"].extend(clauses)
                turn_audio_buffers[-1].append(sub_audio)
                if eng_meaning:
                    prev["english_translation"] = (
                        f"{prev.get('english_translation', '')} {eng_meaning}".strip()
                    )
                existing_codes = {l["code"] for l in prev["languages"]}
                for lc in lang_codes:
                    if lc not in existing_codes:
                        prev["languages"].append({"code": lc, "name": get_language_name(lc)})
                        existing_codes.add(lc)
                prev["language_switched"] = len(prev["languages"]) > 1
            else:
                turn_id = f"turn_{int(time.time() * 1000)}_{len(turns) + 1}"
                turns.append(
                    {
                        "id": turn_id,
                        "speaker_id": spk_info["id"],
                        "speaker_label": spk_info["label"],
                        "speaker_color": spk_info["color"],
                        "speaker_index": spk_info["index"],
                        "speaker_confidence": spk_conf,
                        "start": round(s_samp / float(SAMPLE_RATE), 2),
                        "end": round(e_samp / float(SAMPLE_RATE), 2),
                        "languages": [
                            {"code": c, "name": get_language_name(c)} for c in lang_codes
                        ],
                        "language_switched": len(lang_codes) > 1,
                        "clauses": clauses,
                        "text": text,
                        "english_translation": eng_meaning,
                    }
                )
                turn_audio_buffers.append([sub_audio])

        # Attach playable WAV base64 and save WAV file to backend/recordings/ for each turn
        for idx, turn_obj in enumerate(turns):
            try:
                joined_audio = np.concatenate(turn_audio_buffers[idx])
                wav_bytes = float32_to_wav_bytes(joined_audio, SAMPLE_RATE)
                turn_obj["audio_wav_b64"] = base64.b64encode(wav_bytes).decode("utf-8")
                rec_filename = f"{turn_obj['id']}.wav"
                rec_path = os.path.join(RECORDINGS_DIR, rec_filename)
                with open(rec_path, "wb") as wf:
                    wf.write(wav_bytes)
                turn_obj["recording_file"] = f"backend/recordings/{rec_filename}"
            except Exception as e:
                print(f"[Recordings] Warning saving turn audio: {e}", flush=True)

        proc_time = round(time.time() - t0, 2)
        return {
            "turns": turns,
            "speakers": self.diarizer.get_speakers_summary(),
            "duration": total_duration,
            "processing_time": proc_time,
            "engine": engine_mode,
        }


# Singleton instance used by server.py
transcriber_engine = MultilingualTranscriber()


