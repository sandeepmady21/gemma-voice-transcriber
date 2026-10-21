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

import base64
from collections import OrderedDict
import http.server
import io
import json
import os
import platform
import re
import socket
import socketserver
import ssl
import subprocess
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np

from transcriber import LANGUAGE_NAMES, SAMPLE_RATE, transcriber_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_STT_LANGS = {"en", "ar", "es", "ja", "zh", "ko"}
MAX_MODELS = 2
_stt_recognizers = OrderedDict()
_stt_lock = threading.RLock()

TTS_LANG_MAP = {
    "ar": "ar-msa",
    "en": "en-us",
    "es": "es-es",
    "ja": "ja-jp",
    "zh": "zh-hans",
    "ko": "ko-kr",
}
TTS_VOICE_MAP = {
    "zh": "kokoro_zf_xiaoxiao",
}
_tts_engines = OrderedDict()
_tts_lock = threading.RLock()


def get_tts_engine(language="en"):
    if language not in TTS_LANG_MAP:
        language = "en"
    with _tts_lock:
        if language in _tts_engines:
            _tts_engines.move_to_end(language)
            return _tts_engines[language]
        from moonshine_voice import TextToSpeech

        moon_lang = TTS_LANG_MAP[language]
        voice = TTS_VOICE_MAP.get(language)
        print(
            f"[TTS] Loading moonshine-voice (lang={language} -> {moon_lang}, voice={voice or 'default'})..."
        )
        if len(_tts_engines) >= MAX_MODELS:
            oldest_lang, oldest_engine = _tts_engines.popitem(last=False)
            print(f"[TTS] Evicting model for {oldest_lang}")
            del oldest_engine
        if voice:
            _tts_engines[language] = TextToSpeech(moon_lang, voice=voice)
        else:
            _tts_engines[language] = TextToSpeech(moon_lang)
        return _tts_engines[language]


def get_stt_recognizer(language="en"):
    if language not in SUPPORTED_STT_LANGS:
        language = "en"
    with _stt_lock:
        if language in _stt_recognizers:
            _stt_recognizers.move_to_end(language)
            return _stt_recognizers[language]
        from moonshine_voice import Transcriber, get_model_for_language

        print(f"[STT] Loading Moonshine STT (lang={language})...")
        if len(_stt_recognizers) >= MAX_MODELS:
            oldest_lang, oldest_recognizer = _stt_recognizers.popitem(last=False)
            print(f"[STT] Evicting model for {oldest_lang}")
            del oldest_recognizer
        model_path, model_arch = get_model_for_language(language)
        _stt_recognizers[language] = Transcriber(
            model_path=model_path, model_arch=model_arch
        )
        return _stt_recognizers[language]


PORT = 3000


class ProxyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "X-Requested-With, Content-Type, x-target-url, authorization",
        )
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_proxy(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        target_url = query.get("url", [None])[0]

        if not target_url:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Error: Missing "url" query parameter.')
            return

        parsed_target = urllib.parse.urlparse(target_url)
        if (
            parsed_target.scheme not in ("http", "https")
            or parsed_target.hostname not in ("localhost", "127.0.0.1")
            or parsed_target.port not in (9379, 11434, 1234, None)
        ):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(
                b"Forbidden: Proxy target must be a local LLM port (9379, 11434, 1234)"
            )
            return

        print(f"[Proxy] Routing {self.command} request to: {target_url}")

        body = None
        if self.command in ["POST", "PUT", "PATCH"]:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

        req = urllib.request.Request(target_url, data=body, method=self.command)

        for key, val in self.headers.items():
            if key.lower() not in [
                "host",
                "connection",
                "content-length",
                "x-target-url",
            ]:
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                res_body = response.read()
                self.send_response(response.status)
                for key, val in response.headers.items():
                    if key.lower() not in ["content-length", "connection"]:
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(res_body)
        except urllib.error.HTTPError as e:
            print(f"[Proxy Error] HTTP Error {e.code}: {e.reason}")
            try:
                res_body = e.read()
            except Exception:
                res_body = str(e).encode("utf-8")
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(res_body)
        except Exception as e:
            print(f"[Proxy Error] Exception: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))

    def handle_transcribe(self):
        """
        Multilingual code-switching & speaker-diarized transcription endpoint.
        POST /api/transcribe
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            if not body:
                raise ValueError("No request body")

            data = json.loads(body.decode("utf-8"))
            audio_b64 = data.get("audio_base64")
            if not audio_b64:
                raise ValueError("Missing audio_base64 parameter")

            engine_mode = data.get("engine", "gemma4-audio")
            whisper_model_size = data.get("whisper_model", "small")
            allowed_languages = data.get("allowed_languages") or None
            speaker_threshold = data.get("speaker_threshold")
            if speaker_threshold is not None:
                speaker_threshold = float(speaker_threshold)

            endpoint_url = data.get("endpoint_url", "http://localhost:9379/v1").rstrip("/")
            if not endpoint_url.endswith("/v1"):
                endpoint_url += "/v1"
            litert_url = f"{endpoint_url}/chat/completions"
            gemma_model = data.get("model_name", "gemma4-e2b")

            raw_data = base64.b64decode(audio_b64)
            audio_np = np.frombuffer(raw_data, dtype=np.float32)

            result = transcriber_engine.transcribe_conversation_chunk(
                audio_np=audio_np,
                engine_mode=engine_mode,
                whisper_model_size=whisper_model_size,
                allowed_languages=allowed_languages,
                speaker_threshold=speaker_threshold,
                litert_url=litert_url,
                gemma_model=gemma_model,
            )
            print(
                f"[Transcribe] Processed {result['duration']}s in {result['processing_time']}s "
                f"(turns={len(result['turns'])}, speakers={len(result['speakers'])}, engine={engine_mode})"
            )
            self._send_json(200, result)
        except Exception as e:
            traceback.print_exc()
            print(f"[Transcribe Error] {e}")
            self._send_json(500, {"error": str(e)})

    def handle_speakers(self):
        """
        GET /api/speakers -> returns active speakers in session
        POST /api/speakers -> rename, enroll, or reset speakers
        """
        try:
            if self.command == "GET":
                self._send_json(
                    200, {"speakers": transcriber_engine.diarizer.get_speakers_summary()}
                )
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(body.decode("utf-8"))
            action = data.get("action", "list")

            if action == "reset":
                transcriber_engine.diarizer.reset_speakers()
                self._send_json(200, {"status": "ok", "speakers": []})
            elif action == "rename":
                speaker_id = data.get("speaker_id", "")
                new_label = data.get("label", "")
                ok = transcriber_engine.diarizer.rename_speaker(speaker_id, new_label)
                self._send_json(
                    200 if ok else 404,
                    {
                        "status": "ok" if ok else "not_found",
                        "speakers": transcriber_engine.diarizer.get_speakers_summary(),
                    },
                )
            elif action == "enroll":
                label = data.get("label", "").strip()
                audio_b64 = data.get("audio_base64", "")
                if not label or not audio_b64:
                    self._send_json(400, {"error": "Missing label or audio_base64"})
                    return
                audio_np = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
                spk = transcriber_engine.diarizer.enroll_speaker(
                    label=label, audio=audio_np, sr=SAMPLE_RATE
                )
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "enrolled_speaker": spk,
                        "speakers": transcriber_engine.diarizer.get_speakers_summary(),
                    },
                )
            else:
                self._send_json(
                    200, {"speakers": transcriber_engine.diarizer.get_speakers_summary()}
                )
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})

    def handle_languages(self):
        """GET /api/languages -> returns all 99+ supported languages."""
        langs = [{"code": k, "name": v} for k, v in LANGUAGE_NAMES.items()]
        self._send_json(200, {"languages": langs})

    def handle_tts(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        text = query.get("text", [None])[0]
        lang = query.get("lang", ["en"])[0]

        if not text:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Error: Missing "text" parameter.')
            return

        print(f"[TTS] Synthesizing with moonshine-voice: {text[:50]}... (lang: {lang})")

        try:
            with _tts_lock:
                engine = get_tts_engine(lang)
                audio, sample_rate = engine.synthesize(text)

            samples = np.asarray(audio, dtype=np.float32)
            samples = np.clip(samples, -1.0, 1.0)
            pcm16 = (samples * 32767.0).astype("<i2")

            with io.BytesIO() as buf:
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(sample_rate))
                    wf.writeframes(pcm16.tobytes())
                wav_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(f"[TTS Error] Exception: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))

    def handle_stt(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            if not body:
                raise ValueError("No body data")

            data = json.loads(body.decode("utf-8"))
            audio_b64 = data.get("audio_base64")
            if not audio_b64:
                raise ValueError("Missing audio_base64 parameter")

            language = data.get("language", "en")
            raw_data = base64.b64decode(audio_b64)
            audio_np = np.frombuffer(raw_data, dtype=np.float32)

            with _stt_lock:
                recognizer = get_stt_recognizer(language)
                transcript = recognizer.transcribe_without_streaming(audio_np, 16000)
            text = " ".join([line.text for line in transcript.lines])
            print(f"[STT] Transcribed: {text}")

            self._send_json(200, {"text": text})
        except Exception as e:
            traceback.print_exc()
            print(f"[STT Error] Exception: {e}")
            self._send_json(500, {"error": str(e)})

    def handle_volume(self):
        client_ip = self.client_address[0]
        if client_ip not in ("127.0.0.1", "::1", "localhost"):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"Forbidden: Volume control is only accessible locally")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8")) if body else {}
                action = data.get("action")
            else:
                action = "get"

            env = os.environ.copy()
            if "XDG_RUNTIME_DIR" not in env:
                uid = os.getuid()
                env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"

            is_mac = platform.system() == "Darwin"

            def get_vol():
                if is_mac:
                    try:
                        out = subprocess.check_output(
                            ["osascript", "-e", "output volume of (get volume settings)"],
                            text=True,
                            timeout=2,
                        ).strip()
                        return int(out)
                    except Exception:
                        pass
                try:
                    out = subprocess.check_output(
                        ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                        text=True,
                        timeout=2,
                        env=env,
                    )
                    m = re.search(r"Volume:\s+([0-9.]+)", out)
                    if m:
                        return round(float(m.group(1)) * 100)
                except Exception:
                    pass
                try:
                    out = subprocess.check_output(
                        ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                        text=True,
                        timeout=2,
                        env=env,
                    )
                    m = re.search(r"(\d+)%", out)
                    if m:
                        return int(m.group(1))
                except Exception:
                    pass
                try:
                    out = subprocess.check_output(
                        ["amixer", "sget", "Master"],
                        text=True,
                        timeout=2,
                        env=env,
                    )
                    m = re.search(r"\[(\d+)%\]", out)
                    if m:
                        return int(m.group(1))
                except Exception:
                    pass
                return None

            def set_vol(direction):
                if is_mac:
                    try:
                        cur = get_vol()
                        if cur is None:
                            cur = 50
                        target = min(100, cur + 5) if direction == "up" else max(0, cur - 5)
                        subprocess.run(
                            ["osascript", "-e", f"set volume output volume {target}"],
                            check=True,
                            timeout=2,
                        )
                        return True
                    except Exception:
                        pass
                try:
                    arg = "5%+" if direction == "up" else "5%-"
                    subprocess.run(
                        ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", arg],
                        check=True,
                        timeout=2,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception:
                    pass
                try:
                    arg = "+5%" if direction == "up" else "-5%"
                    subprocess.run(
                        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", arg],
                        check=True,
                        timeout=2,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception:
                    pass
                try:
                    arg = "5%+" if direction == "up" else "5%-"
                    subprocess.run(
                        ["amixer", "sset", "Master", arg],
                        check=True,
                        timeout=2,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception:
                    pass
                return False

            success = False
            if action in ("up", "down"):
                success = set_vol(action)
            elif action == "get":
                success = True

            if success:
                current_vol = get_vol()
                self._send_json(200, {"status": "ok", "volume": current_vol})
            else:
                self._send_json(500, {"error": "Failed to change system volume"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_POST(self):
        if self.path.startswith("/proxy"):
            self.handle_proxy()
            return
        if self.path.startswith("/api/transcribe"):
            self.handle_transcribe()
            return
        if self.path.startswith("/api/speakers"):
            self.handle_speakers()
            return
        if self.path.startswith("/api/stt"):
            self.handle_stt()
            return
        if self.path.startswith("/api/volume"):
            self.handle_volume()
            return

        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/proxy"):
            self.handle_proxy()
            return
        if self.path.startswith("/api/speakers"):
            self.handle_speakers()
            return
        if self.path.startswith("/api/languages"):
            self.handle_languages()
            return
        if self.path.startswith("/api/tts"):
            self.handle_tts()
            return
        if self.path.startswith("/api/volume"):
            self.handle_volume()
            return

        url_path = self.path.split("?", 1)[0]
        if url_path == "/":
            url_path = "/index.html"

        dist_dir = os.path.realpath(os.path.join(BASE_DIR, "..", "frontend", "dist"))
        if not os.path.exists(dist_dir):
            dist_dir = os.path.realpath(os.path.join(BASE_DIR, "dist"))
        if not os.path.exists(dist_dir):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"dist/ directory not found")
            return

        filename = url_path.lstrip("/")
        filepath = os.path.realpath(os.path.join(dist_dir, filename))

        if not filepath.startswith(dist_dir + os.sep) and filepath != dist_dir:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"Forbidden")
            return

        if not os.path.exists(filepath) or os.path.isdir(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return

        ext = os.path.splitext(filepath)[1].lower()
        mime_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        content_type = mime_types.get(ext, "application/octet-stream")

        try:
            with open(filepath, "rb") as f:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(f.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    local_ip = "localhost"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    use_ssl = os.path.exists("cert.pem") and os.path.exists("key.pem")

    with socketserver.ThreadingTCPServer(("", PORT), ProxyHTTPRequestHandler) as httpd:
        if use_ssl:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

        protocol = "https" if use_ssl else "http"
        print("===========================================================")
        print("Gemma Offline Multilingual Transcriber & Diarizer running at:")
        print(f"👉 {protocol}://localhost:{PORT}")
        if local_ip != "localhost":
            print(f"👉 {protocol}://{local_ip}:{PORT} (Local Network)")
        print("===========================================================")

        def _prewarm_models():
            try:
                print(
                    "[Prewarm] Loading neural speaker diarizer & multilingual ASR models...",
                    flush=True,
                )
                transcriber_engine.diarizer._ensure_extractor()
                transcriber_engine.get_whisper_model("small")
                print("[Prewarm] Diarizer & ASR models pre-warmed successfully.", flush=True)
            except Exception as e:
                print(f"[Prewarm Warning] {e}", flush=True)

        threading.Thread(target=_prewarm_models, daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
