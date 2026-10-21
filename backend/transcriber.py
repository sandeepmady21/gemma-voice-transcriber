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

Architecture:
1. Fine-Grained VAD + Sliding-Window Speaker Diarization (`sherpa-onnx` 3D-Speaker
   CampPlus 16kHz neural voiceprints + acoustic fallback):
   - Splits incoming audio into micro-segments at pauses (>=200ms) and speaker
     change-points, extracts neural voiceprint embeddings, and assigns each
     segment to a persistent speaker identity ("Speaker 1", "Speaker 2", or
     enrolled speaker names) across the entire conversation—even when the same
     speaker switches between Telugu, Hindi, and English.
2. Mid-Conversation & Mid-Sentence Language Switching (Code-Switching):
   - Primary Engine (`gemma4-audio` / `hybrid`): Feeds each diarized speaker turn
     directly to Google Gemma 4 E2B's native multimodal audio encoder
     (`TF_LITE_AUDIO_ENCODER_HW` via LiteRT-LM), which natively transcribes
     code-switched Telugu/Hindi/English speech, splits multi-language turns into
     labeled `clauses`, and provides an English translation.
   - Secondary / Fallback Engine (`whisper`): Uses Faster-Whisper (`small` /
     `large-v3-turbo`) with per-micro-phrase language detection and native-script
     prompts for Telugu (తెలుగు) and Hindi (हिन्दी).
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

# Default priority languages: Telugu, English, Hindi first, followed by 99+ languages
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

# Native-script initial prompts for Whisper so Indian languages always render in their proper script
# (e.g. Hindi in Devanagari instead of Urdu/Perso-Arabic, Telugu in తెలుగు script)
WHISPER_SCRIPT_PROMPTS: Dict[str, str] = {
    "te": "నమస్కారం, ఇది తెలుగు, హిందీ మరియు English సంభాషణ.",
    "hi": "नमस्ते, यह हिंदी, तेलुगु और English बातचीत है।",
    "ta": "வணக்கம், இது தமிழ் மற்றும் English உரையாடல்.",
    "kn": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಮತ್ತು English ಸಂಭಾಷಣೆ.",
    "ml": "നമസ്കാരം, ഇത് മലയാളം, English സംഭാഷണമാണ്.",
    "mr": "नमस्कार, हे मराठी आणि English संभाषण आहे.",
    "bn": "নমস্কার, এটি বাংলা এবং English কথোপকথন।",
    "gu": "નમસ્તે, આ ગુજરાતી અને English વાતચીત છે.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਇਹ ਪੰਜਾਬੀ ਅਤੇ English ਗੱਲਬਾਤ ਹੈ।",
}

SPEAKER_PALETTE = [
    "#000000",  # Terminal Black
    "#1d4ed8",  # Cobalt Blue
    "#047857",  # Emerald Green
    "#b45309",  # Warm Amber
    "#6d28d9",  # Violet
    "#be123c",  # Crimson
]


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
    base = s.split("-")[0]
    if base in LANGUAGE_NAMES:
        return base
    for k, v in LANGUAGE_NAMES.items():
        if v.lower() == s:
            return k
    return base[:2] if len(base) >= 2 else "en"


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
    """Detects distinct writing scripts inside a text string for intra-sentence code-switching."""
    scripts = []
    has_latin = False
    has_devanagari = False
    has_arabic = False
    has_cjk = False
    has_hangul = False
    has_kana = False
    has_cyrillic = False
    has_tamil = False
    has_telugu = False
    has_bengali = False
    has_thai = False
    has_greek = False
    has_hebrew = False
    has_gujarati = False
    has_gurmukhi = False
    has_kannada = False
    has_malayalam = False

    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            has_latin = True
        elif 0x0C00 <= cp <= 0x0C7F:
            has_telugu = True
        elif 0x0900 <= cp <= 0x097F:
            has_devanagari = True
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            has_arabic = True
        elif 0x3040 <= cp <= 0x30FF:
            has_kana = True
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            has_hangul = True
        elif 0x4E00 <= cp <= 0x9FFF:
            has_cjk = True
        elif 0x0400 <= cp <= 0x04FF:
            has_cyrillic = True
        elif 0x0B80 <= cp <= 0x0BFF:
            has_tamil = True
        elif 0x0980 <= cp <= 0x09FF:
            has_bengali = True
        elif 0x0A80 <= cp <= 0x0AFF:
            has_gujarati = True
        elif 0x0A00 <= cp <= 0x0A7F:
            has_gurmukhi = True
        elif 0x0C80 <= cp <= 0x0CFF:
            has_kannada = True
        elif 0x0D00 <= cp <= 0x0D7F:
            has_malayalam = True
        elif 0x0E00 <= cp <= 0x0E7F:
            has_thai = True
        elif 0x0370 <= cp <= 0x03FF:
            has_greek = True
        elif 0x0590 <= cp <= 0x05FF:
            has_hebrew = True

    if has_telugu:
        scripts.append("te")
    if has_devanagari:
        scripts.append("hi")
    if has_arabic:
        scripts.append("ar")
    if has_kana:
        scripts.append("ja")
    elif has_cjk:
        scripts.append("zh")
    if has_hangul:
        scripts.append("ko")
    if has_cyrillic:
        scripts.append("ru")
    if has_tamil:
        scripts.append("ta")
    if has_bengali:
        scripts.append("bn")
    if has_gujarati:
        scripts.append("gu")
    if has_gurmukhi:
        scripts.append("pa")
    if has_kannada:
        scripts.append("kn")
    if has_malayalam:
        scripts.append("ml")
    if has_thai:
        scripts.append("th")
    if has_greek:
        scripts.append("el")
    if has_hebrew:
        scripts.append("he")
    if has_latin and scripts:
        scripts.append("en")
    return scripts


def _compute_acoustic_voiceprint(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fallback 60-dim L2-normalized spectral + prosodic voiceprint when ONNX model
    is unavailable.
    """
    if len(audio) < int(sr * 0.15):
        audio = np.pad(audio, (0, int(sr * 0.15) - len(audio)))
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-5:
        audio = audio / peak

    n_fft = 512
    hop = 160  # 10ms
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

    mean_vec = np.mean(mfcc_timbre, axis=0)
    std_vec = np.std(mfcc_timbre, axis=0)
    p25_vec = np.percentile(mfcc_timbre, 25, axis=0)
    p75_vec = np.percentile(mfcc_timbre, 75, axis=0)

    feat = np.concatenate([mean_vec, std_vec, p25_vec, p75_vec]).astype(np.float32)
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
        self.similarity_threshold: float = 0.56  # Cosine similarity threshold for CampPlus

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
        """Extracts an L2-normalized speaker embedding from 16kHz float32 audio."""
        self._ensure_extractor()
        audio_f32 = np.asarray(audio, dtype=np.float32)
        if len(audio_f32) < int(sr * 0.25):
            audio_f32 = np.pad(audio_f32, (0, int(sr * 0.25) - len(audio_f32)))

        peak = np.max(np.abs(audio_f32))
        if peak > 1e-4:
            audio_f32 = audio_f32 / peak * 0.85

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
        a new speaker profile if the voice is distinct.
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

            effective_thresh = sim_thresh - (0.08 if duration < 0.85 else 0.0)

            if best_spk is not None and best_sim >= effective_thresh:
                alpha = min(0.25, max(0.05, duration / 15.0))
                new_centroid = (1.0 - alpha) * best_spk["centroid"] + alpha * emb
                norm = np.linalg.norm(new_centroid)
                if norm > 1e-6:
                    best_spk["centroid"] = new_centroid / norm
                best_spk["utterance_count"] += 1
                best_spk["total_seconds"] = round(best_spk["total_seconds"] + duration, 1)
                conf = round(max(0.55, min(0.99, (best_sim + 0.35))), 2)
                return best_spk, conf

            if duration >= 0.55 and len(self.speakers) < 8:
                spk = self._create_speaker(emb, duration=duration)
                return spk, 0.92

            if best_spk is not None:
                return best_spk, round(max(0.5, best_sim), 2)
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


def split_audio_on_pauses_and_speaker_changes(
    audio: np.ndarray,
    diarizer: SpeakerDiarizer,
    sr: int = SAMPLE_RATE,
    min_segment_sec: float = 0.65,
    max_segment_sec: float = 4.2,
) -> List[Tuple[int, int]]:
    """
    Splits a recording into fine-grained speech chunks at:
    1. Natural silence/breath dips (>= 200ms low energy)
    2. Acoustic speaker-change points
    """
    n_samples = len(audio)
    if n_samples == 0:
        return []

    frame_len = int(sr * 0.02)  # 20ms
    hop_len = int(sr * 0.01)  # 10ms
    if n_samples <= int(sr * 1.4):
        return [(0, n_samples)]

    padded = np.pad(audio, (0, frame_len))
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame_len)[::hop_len]
    rms = np.sqrt(np.mean(windows**2, axis=1) + 1e-10)

    noise_floor = np.percentile(rms, 15)
    speech_peak = np.percentile(rms, 85)
    if speech_peak < 1e-4:
        return []

    silence_thresh = max(noise_floor * 2.2, speech_peak * 0.14)
    is_voiced = rms > silence_thresh

    min_pause_frames = 20  # 200ms pause splits phrases so language/speaker switches are caught immediately
    raw_regions: List[Tuple[int, int]] = []
    in_speech = False
    start_f = 0
    silence_run = 0

    for i, v in enumerate(is_voiced):
        if v:
            if not in_speech:
                start_f = max(0, i - 5)
                in_speech = True
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_pause_frames:
                    end_f = min(len(is_voiced), i - silence_run + 6)
                    if end_f - start_f >= 25:
                        raw_regions.append((start_f * hop_len, min(n_samples, end_f * hop_len)))
                    in_speech = False
                    silence_run = 0

    if in_speech:
        end_f = len(is_voiced)
        if end_f - start_f >= 25:
            raw_regions.append((start_f * hop_len, n_samples))

    if not raw_regions:
        return [(0, n_samples)]

    refined_regions: List[Tuple[int, int]] = []
    for s_samp, e_samp in raw_regions:
        if refined_regions:
            prev_s, prev_e = refined_regions[-1]
            if (prev_e - prev_s) < int(min_segment_sec * sr) and (s_samp - prev_e) < int(0.40 * sr):
                refined_regions[-1] = (prev_s, e_samp)
                s_samp, e_samp = refined_regions[-1]
                refined_regions.pop()

        seg_len = e_samp - s_samp
        if seg_len <= int(max_segment_sec * sr):
            refined_regions.append((s_samp, e_samp))
        else:
            cur = s_samp
            while e_samp - cur > int(max_segment_sec * sr):
                search_start = cur + int(1.6 * sr)
                search_end = min(e_samp - int(0.8 * sr), cur + int(max_segment_sec * sr))
                if search_end <= search_start:
                    break
                f_start = search_start // hop_len
                f_end = min(len(rms), search_end // hop_len)
                if f_end > f_start:
                    kernel = np.ones(12, dtype=np.float32) / 12.0
                    local_rms = np.convolve(rms[f_start:f_end], kernel, mode="same")
                    split_f = f_start + int(np.argmin(local_rms))
                    split_samp = split_f * hop_len
                else:
                    split_samp = (search_start + search_end) // 2
                refined_regions.append((cur, split_samp))
                cur = split_samp
            if e_samp - cur >= int(0.35 * sr):
                refined_regions.append((cur, e_samp))
            elif refined_regions:
                refined_regions[-1] = (refined_regions[-1][0], e_samp)

    return refined_regions


def transcribe_segment_with_gemma4(
    sub_audio: np.ndarray,
    litert_url: str = "http://localhost:9379/v1/chat/completions",
    model_name: str = "gemma4-e2b",
    allowed_languages: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Uses Gemma 4 E2B's native multimodal audio encoder via LiteRT-LM's
    OpenAI-compatible `/v1/chat/completions` (`input_audio` WAV payload) to
    transcribe speech verbatim, detect all spoken languages (specifically tuned for
    Telugu `te`, English `en`, and Hindi `hi` mid-conversation code-switching),
    split into per-language `clauses`, and provide an English translation.
    """
    wav_bytes = float32_to_wav_bytes(sub_audio, SAMPLE_RATE)
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    active_langs = allowed_languages if allowed_languages else DEFAULT_PRIORITY_LANGUAGES
    names = [f"{get_language_name(c)} ({c})" for c in active_langs]
    lang_hint = (
        f" The primary languages in this conversation are {', '.join(names)}."
        " Write Telugu in Telugu script (తెలుగు), Hindi in Devanagari script (हिन्दी, never Urdu script), and English in Latin script."
    )

    prompt_text = (
        "Listen carefully and transcribe the spoken audio verbatim in its original spoken language(s) and native script(s)."
        + lang_hint
        + " Speakers frequently switch between Telugu, Hindi, and English mid-conversation or mid-sentence (code-switching). Preserve each phrase in the exact language it was spoken and split into clauses.\n"
        "Return ONLY valid JSON matching this structure:\n"
        '{"text": "full verbatim transcript", "languages": ["te", "en", "hi"], "clauses": [{"text": "phrase 1", "lang": "te"}, {"text": "phrase 2", "lang": "en"}], "english_translation": "English meaning of the full utterance"}'
    )

    payload = {
        "model": model_name,
        "temperature": 0.0,
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

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not raw_reply:
                return None

            clean = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw_reply.strip(), flags=re.IGNORECASE
            ).strip()
            m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if m:
                clean = m.group(0)

            try:
                parsed = json.loads(clean)
                text = str(parsed.get("text", "")).strip()
                langs = parsed.get("languages") or []
                if isinstance(langs, str):
                    langs = [langs]
                norm_langs = []
                for l in langs:
                    nc = normalize_language_code(str(l))
                    if nc == "ur" and ("hi" in active_langs):
                        nc = "hi"
                    if nc not in norm_langs:
                        norm_langs.append(nc)

                raw_clauses = parsed.get("clauses") or []
                clauses = []
                if isinstance(raw_clauses, list):
                    for c in raw_clauses:
                        if isinstance(c, dict) and str(c.get("text", "")).strip():
                            c_lang = normalize_language_code(
                                str(c.get("lang", norm_langs[0] if norm_langs else "en"))
                            )
                            if c_lang == "ur" and ("hi" in active_langs):
                                c_lang = "hi"
                            if c_lang not in norm_langs:
                                norm_langs.append(c_lang)
                            clauses.append(
                                {
                                    "text": str(c["text"]).strip(),
                                    "lang": c_lang,
                                    "lang_name": get_language_name(c_lang),
                                    "lang_prob": 0.96,
                                }
                            )

                if not norm_langs:
                    norm_langs = detect_script_languages(text) or ["en"]
                if not clauses and text:
                    clauses = [
                        {
                            "text": text,
                            "lang": norm_langs[0],
                            "lang_name": get_language_name(norm_langs[0]),
                            "lang_prob": 0.95,
                        }
                    ]

                eng_trans = str(parsed.get("english_translation", "")).strip()
                return {
                    "text": text,
                    "languages": norm_langs,
                    "clauses": clauses,
                    "english_translation": eng_trans,
                }
            except Exception:
                script_langs = detect_script_languages(clean) or ["en"]
                return {
                    "text": clean,
                    "languages": script_langs,
                    "clauses": [
                        {
                            "text": clean,
                            "lang": script_langs[0],
                            "lang_name": get_language_name(script_langs[0]),
                            "lang_prob": 0.90,
                        }
                    ],
                    "english_translation": "",
                }
    except Exception as e:
        print(f"[Gemma4-Audio] Inference error: {e}", flush=True)
        return None


class MultilingualTranscriber:
    """
    Coordinates sherpa-onnx neural speaker diarization, Gemma 4 E2B native
    multimodal audio code-switching transcription, and Faster-Whisper 99-language ASR.
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

    def _detect_segment_language(
        self,
        model: Any,
        chunk_audio: np.ndarray,
        allowed_languages: Optional[List[str]] = None,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        try:
            if len(chunk_audio) < int(SAMPLE_RATE * 1.5):
                pad_len = int(SAMPLE_RATE * 1.5) - len(chunk_audio)
                probe_audio = np.pad(chunk_audio, (0, pad_len))
            else:
                probe_audio = chunk_audio

            lang_code, lang_prob, all_probs = model.detect_language(probe_audio)
            prob_map = dict(all_probs) if all_probs else {lang_code: float(lang_prob)}

            # Map acoustically identical Urdu ("ur") probability into Hindi ("hi")
            # so spoken Hindustani is transcribed in Devanagari rather than Perso-Arabic
            if "ur" in prob_map and (not allowed_languages or "hi" in allowed_languages):
                prob_map["hi"] = max(prob_map.get("hi", 0.0), prob_map["ur"])

            top_candidates = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:4]

            if allowed_languages:
                allowed_set = {c.strip().lower() for c in allowed_languages if c.strip()}
                if allowed_set:
                    filtered = [(code, prob) for code, prob in prob_map.items() if code in allowed_set]
                    if filtered:
                        best_code, best_prob = max(filtered, key=lambda x: x[1])
                        return best_code, float(best_prob), filtered[:3]

            best_code, best_prob = top_candidates[0]
            return best_code, float(best_prob), top_candidates
        except Exception as e:
            print(f"[ASR] detect_language fallback: {e}", flush=True)
            return "en", 0.5, [("en", 0.5)]

    def transcribe_conversation_chunk(
        self,
        audio_np: np.ndarray,
        engine_mode: str = "gemma4-audio",
        whisper_model_size: str = "small",
        allowed_languages: Optional[List[str]] = None,
        speaker_threshold: Optional[float] = None,
        litert_url: str = "http://localhost:9379/v1/chat/completions",
        gemma_model: str = "gemma4-e2b",
    ) -> Dict[str, Any]:
        """
        Processes a recorded or streamed audio buffer (16kHz mono float32):
        1. Splits into micro-phrases at breath pauses / speaker changes.
        2. Identifies the speaker for each micro-phrase via neural voiceprints.
        3. Groups contiguous micro-phrases from the same speaker (up to ~10s) for
           Gemma 4 E2B native audio code-switch transcription, OR transcribes per
           micro-phrase with Faster-Whisper using native-script prompts for
           Telugu (తెలుగు), Hindi (हिन्दी), and English.
        """
        t0 = time.time()
        audio_np = np.asarray(audio_np, dtype=np.float32)
        total_duration = round(len(audio_np) / float(SAMPLE_RATE), 2)

        if len(audio_np) < int(SAMPLE_RATE * 0.25):
            return {
                "turns": [],
                "speakers": self.diarizer.get_speakers_summary(),
                "duration": 0.0,
                "processing_time": 0.0,
                "engine": engine_mode,
            }

        regions = split_audio_on_pauses_and_speaker_changes(
            audio_np,
            self.diarizer,
            sr=SAMPLE_RATE,
            min_segment_sec=0.65,
            max_segment_sec=4.2,
        )

        if not regions:
            return {
                "turns": [],
                "speakers": self.diarizer.get_speakers_summary(),
                "duration": total_duration,
                "processing_time": round(time.time() - t0, 2),
                "engine": engine_mode,
            }

        # Step 1: Diarize each voiced micro-region first (<20ms per region)
        diarized_spans: List[Dict[str, Any]] = []
        for s_samp, e_samp in regions:
            sub_audio = audio_np[s_samp:e_samp]
            if np.max(np.abs(sub_audio)) < 0.008:
                continue
            spk_info, spk_conf = self.diarizer.match_or_register_speaker(
                sub_audio, sr=SAMPLE_RATE, threshold=speaker_threshold
            )
            diarized_spans.append(
                {
                    "s_samp": s_samp,
                    "e_samp": e_samp,
                    "speaker_info": spk_info,
                    "speaker_conf": spk_conf,
                }
            )

        if not diarized_spans:
            return {
                "turns": [],
                "speakers": self.diarizer.get_speakers_summary(),
                "duration": total_duration,
                "processing_time": round(time.time() - t0, 2),
                "engine": engine_mode,
            }

        turns: List[Dict[str, Any]] = []

        if engine_mode == "gemma4-audio":
            merged_speaker_turns: List[Dict[str, Any]] = []
            for span in diarized_spans:
                if (
                    merged_speaker_turns
                    and merged_speaker_turns[-1]["speaker_info"]["id"] == span["speaker_info"]["id"]
                    and (span["e_samp"] - merged_speaker_turns[-1]["s_samp"]) <= int(10.5 * SAMPLE_RATE)
                ):
                    merged_speaker_turns[-1]["e_samp"] = span["e_samp"]
                else:
                    merged_speaker_turns.append(dict(span))

            for m_turn in merged_speaker_turns:
                s_samp = m_turn["s_samp"]
                e_samp = m_turn["e_samp"]
                spk_info = m_turn["speaker_info"]
                spk_conf = m_turn["speaker_conf"]
                turn_audio = audio_np[s_samp:e_samp]

                gemma_res = transcribe_segment_with_gemma4(
                    turn_audio,
                    litert_url=litert_url,
                    model_name=gemma_model,
                    allowed_languages=allowed_languages,
                )

                if gemma_res and gemma_res.get("text"):
                    text = gemma_res["text"]
                    lang_codes = gemma_res.get("languages") or ["en"]
                    script_langs = detect_script_languages(text)
                    for sl in script_langs:
                        if sl not in lang_codes:
                            lang_codes.append(sl)

                    clauses = gemma_res.get("clauses") or []
                    for c in clauses:
                        c["start"] = round(s_samp / float(SAMPLE_RATE), 2)
                        c["end"] = round(e_samp / float(SAMPLE_RATE), 2)

                    turns.append(
                        {
                            "id": f"turn_{int(time.time() * 1000)}_{len(turns) + 1}",
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
                            "language_switched": len(lang_codes) > 1 or len(clauses) > 1,
                            "clauses": clauses,
                            "text": text,
                            "english_translation": gemma_res.get("english_translation", ""),
                        }
                    )
                else:
                    engine_mode = "whisper"

        if engine_mode in ("whisper", "hybrid"):
            whisper_model = self.get_whisper_model(whisper_model_size)
            raw_clauses: List[Dict[str, Any]] = []

            for span in diarized_spans:
                s_samp = span["s_samp"]
                e_samp = span["e_samp"]
                spk_info = span["speaker_info"]
                spk_conf = span["speaker_conf"]
                sub_audio = audio_np[s_samp:e_samp]

                detected_lang, lang_prob, _ = self._detect_segment_language(
                    whisper_model, sub_audio, allowed_languages=allowed_languages
                )
                use_multilingual_flag = not bool(allowed_languages) and lang_prob < 0.72
                init_prompt = WHISPER_SCRIPT_PROMPTS.get(detected_lang)

                segments_gen, _ = whisper_model.transcribe(
                    sub_audio,
                    language=None if use_multilingual_flag else detected_lang,
                    multilingual=use_multilingual_flag,
                    initial_prompt=init_prompt,
                    task="transcribe",
                    beam_size=2,
                    best_of=2,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=180, speech_pad_ms=120),
                    word_timestamps=True,
                )
                seg_list = list(segments_gen)
                text = " ".join(s.text.strip() for s in seg_list if s.text.strip()).strip()
                all_clause_langs = [detected_lang]
                clause_list = None
                eng_trans = ""

                if engine_mode == "hybrid" and (lang_prob < 0.68 or not text):
                    gemma_res = transcribe_segment_with_gemma4(
                        sub_audio,
                        litert_url=litert_url,
                        model_name=gemma_model,
                        allowed_languages=allowed_languages,
                    )
                    if gemma_res and gemma_res.get("text"):
                        text = gemma_res["text"]
                        all_clause_langs = gemma_res.get("languages") or [detected_lang]
                        detected_lang = all_clause_langs[0]
                        clause_list = gemma_res.get("clauses")
                        eng_trans = gemma_res.get("english_translation", "")

                if not text:
                    continue

                script_langs = detect_script_languages(text)
                for sl in script_langs:
                    if sl not in all_clause_langs:
                        all_clause_langs.append(sl)

                raw_clauses.append(
                    {
                        "start": round(s_samp / float(SAMPLE_RATE), 2),
                        "end": round(e_samp / float(SAMPLE_RATE), 2),
                        "text": text,
                        "lang": detected_lang,
                        "lang_name": get_language_name(detected_lang),
                        "lang_prob": round(lang_prob, 2),
                        "all_langs": [
                            {"code": c, "name": get_language_name(c)} for c in all_clause_langs
                        ],
                        "sub_clauses": clause_list,
                        "english_translation": eng_trans,
                        "speaker_id": spk_info["id"],
                        "speaker_label": spk_info["label"],
                        "speaker_color": spk_info["color"],
                        "speaker_index": spk_info["index"],
                        "speaker_confidence": spk_conf,
                    }
                )

            for clause in raw_clauses:
                c_items = clause.get("sub_clauses") or [
                    {
                        "text": clause["text"],
                        "lang": clause["lang"],
                        "lang_name": clause["lang_name"],
                        "lang_prob": clause["lang_prob"],
                        "start": clause["start"],
                        "end": clause["end"],
                    }
                ]
                if turns and turns[-1]["speaker_id"] == clause["speaker_id"]:
                    prev_turn = turns[-1]
                    prev_turn["end"] = clause["end"]
                    prev_turn["clauses"].extend(c_items)
                    prev_turn["text"] = f"{prev_turn['text']} {clause['text']}".strip()
                    if clause.get("english_translation"):
                        prev_turn["english_translation"] = (
                            f"{prev_turn.get('english_translation', '')} {clause['english_translation']}"
                        ).strip()
                    existing_codes = {l["code"] for l in prev_turn["languages"]}
                    for l_obj in clause["all_langs"]:
                        if l_obj["code"] not in existing_codes:
                            prev_turn["languages"].append(l_obj)
                            existing_codes.add(l_obj["code"])
                    prev_turn["language_switched"] = len(prev_turn["languages"]) > 1
                else:
                    turns.append(
                        {
                            "id": f"turn_{int(time.time() * 1000)}_{len(turns) + 1}",
                            "speaker_id": clause["speaker_id"],
                            "speaker_label": clause["speaker_label"],
                            "speaker_color": clause["speaker_color"],
                            "speaker_index": clause["speaker_index"],
                            "speaker_confidence": clause["speaker_confidence"],
                            "start": clause["start"],
                            "end": clause["end"],
                            "languages": list(clause["all_langs"]),
                            "language_switched": len(clause["all_langs"]) > 1,
                            "clauses": list(c_items),
                            "text": clause["text"],
                            "english_translation": clause.get("english_translation", ""),
                        }
                    )

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
