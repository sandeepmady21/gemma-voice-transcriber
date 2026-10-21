<img width="960" height="540" src="https://storage.googleapis.com/experiments-uploads/gemma-translator/gemma-translator.gif" />

# Gemma 4 Offline Voice Transcriber & Diarizer (Telugu · English · Hindi + 99 Languages)

Built on top of Google's [Gemma Translator](https://github.com/google-gemma/gemma-translator) and [Google Antigravity](https://antigravity.google/), this project runs an **on-device, 100% offline multilingual voice-to-text transcriber and multi-speaker diarizer** powered by:
- **[Gemma 4 E2B (`gemma4-e2b`)](https://ai.google.dev/gemma/docs/core) + [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-lm)**: Native multimodal audio encoder (`TF_LITE_AUDIO_ENCODER_HW`) that transcribes speech directly from audio, handles mid-sentence code-switching, splits multi-language turns into labeled clauses, and provides live English meaning subtitles.
- **3D-Speaker Neural Voiceprint Diarizer (`sherpa-onnx` CampPlus 16kHz)**: Extracts neural speaker embeddings for every speech segment to automatically identify **which speaker is speaking** (`Speaker 1`, `Speaker 2`, or enrolled speaker names) across the entire conversation—even when the same speaker switches between Telugu, Hindi, and English.
- **Faster-Whisper Multilingual (`small` / `large-v3-turbo`)**: Secondary/hybrid 99-language ASR engine with native-script prompts for **Telugu (`తెలుగు`)** and **Hindi (`हिन्दी`)**.

## Key Features

- **Mid-Conversation & Mid-Sentence Language Switching (Code-Switching)**:
  - Pre-tuned for **Telugu (`te` · తెలుగు)**, **English (`en`)**, and **Hindi (`hi` · हिन्दी)**, with 1-click toggle to **All 99+ Languages**.
  - Splits speech at breath pauses and speaker turns, detects every language spoken inside a turn, and displays clause-by-clause language tags (`TE`, `EN`, `HI`) + an `⚡ CODE-SWITCH` badge.
- **Automatic Speaker Diarization & Voiceprint Enrollment**:
  - Automatically separates and tracks speakers (`Speaker 1`, `Speaker 2`, etc.) using 16kHz neural speaker embeddings.
  - Click any speaker badge to rename them (e.g., `Sandeep`), or use **+ ENROLL VOICE** to register a 3-second voice sample.
- **Dual Recording Modes + Audio File Upload**:
  - **Push-to-Talk (`Hold SPACE` or `Z`)**: Hold to record a turn, release to diarize and transcribe.
  - **Live Auto-Listen (`L` key)**: Hands-free continuous conversation mode with automatic voice activity detection (VAD) chunking.
  - **Audio File Upload (`📁 FILE`)**: Upload `.wav`, `.mp3`, or `.m4a` conversation recordings for offline speaker-diarized transcription.
- **Desktop Console & 480×320 Kiosk Modes**:
  - Toggle between a full-window retro-terminal desktop view (`🖥️ DESKTOP`) and the classic handheld Raspberry Pi viewport (`📟 480×320`).

## Quick Start

1. **Install Dependencies**
   ```bash
   ./setup.sh
   ```

2. **Download the Gemma 4 E2B Model**
   ```bash
   ./download_model.sh
   ```

3. **Start All Local Services**
   ```bash
   ./start.sh
   ```
   - **Web UI (Dev)**: `http://localhost:5173`
   - **API Server / Production UI (`./start.sh --prod`)**: `http://localhost:3000`
   - **LiteRT-LM Server (`gemma4-e2b`)**: `http://localhost:9379`
