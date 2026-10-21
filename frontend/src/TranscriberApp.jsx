/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState, useEffect, useRef, useCallback } from "react"
import Visualizer from "./components/Visualizer"
import { useAudioRecorder } from "./hooks/useAudioRecorder"
import {
  transcribeConversationChunk,
  fetchSpeakers,
  renameSpeakerAPI,
  enrollSpeakerAPI,
  resetSpeakersAPI,
  splitTextIntoSpeechChunks,
} from "./utils/api"
import { playBlip } from "./utils/audio-blip"

const QUICK_LANGUAGES = [
  { code: "te", name: "Telugu", native: "తెలుగు" },
  { code: "en", name: "English", native: "EN" },
  { code: "hi", name: "Hindi", native: "हिन्दी" },
  { code: "ta", name: "Tamil", native: "தமிழ்" },
  { code: "kn", name: "Kannada", native: "ಕನ್ನಡ" },
  { code: "ml", name: "Malayalam", native: "മലയാളം" },
  { code: "mr", name: "Marathi", native: "मराठी" },
  { code: "es", name: "Spanish", native: "ES" },
  { code: "fr", name: "French", native: "FR" },
  { code: "ja", name: "Japanese", native: "日本語" },
  { code: "zh", name: "Chinese", native: "中文" },
  { code: "ar", name: "Arabic", native: "العربية" },
]

export default function TranscriberApp({ config, setConfig }) {
  // Transcript history: array of speaker turns
  const [turns, setTurns] = useState([])
  const [speakers, setSpeakers] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [statusMessage, setStatusMessage] = useState(
    "Ready — Hold SPACE to talk, or enable LIVE AUTO-LISTEN",
  )

  // Language filter state: default to ["te", "en", "hi"] (Telugu, English, Hindi)
  // Empty array [] means "Auto-Detect All 99+ Languages"
  const [selectedLangs, setSelectedLangs] = useState(() => {
    try {
      const saved = localStorage.getItem("transcriberSelectedLangs")
      return saved ? JSON.parse(saved) : ["te", "en", "hi"]
    } catch {
      return ["te", "en", "hi"]
    }
  })

  // Speaker rename / enrollment modal state
  const [editingSpeaker, setEditingSpeaker] = useState(null)
  const [renameValue, setRenameValue] = useState("")
  const [enrollingName, setEnrollingName] = useState("")
  const [showEnrollPanel, setShowEnrollPanel] = useState(false)
  const [isEnrollingRecord, setIsEnrollingRecord] = useState(false)

  const feedEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const audioPlayerRef = useRef(null)

  const {
    isRecording,
    isContinuous,
    startRecording,
    stopRecording,
    decodeAudioFileToBase64,
    analyser,
    micError,
  } = useAudioRecorder()

  useEffect(() => {
    localStorage.setItem(
      "transcriberSelectedLangs",
      JSON.stringify(selectedLangs),
    )
  }, [selectedLangs])

  useEffect(() => {
    fetchSpeakers()
      .then((data) => {
        if (data.speakers) setSpeakers(data.speakers)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (feedEndRef.current) {
      feedEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [turns, isProcessing])

  const processAudioChunk = useCallback(
    async (base64Data) => {
      if (!base64Data) return
      setIsProcessing(true)
      setStatusMessage(
        "Identifying speaker voiceprint & detecting language switches...",
      )

      try {
        const res = await transcribeConversationChunk(base64Data, {
          engine: config.asrEngine || "gemma4-audio",
          whisperModel: config.whisperModel || "small",
          allowedLanguages: selectedLangs,
          speakerThreshold: config.speakerThreshold ?? 0.56,
          endpointUrl: config.endpointUrl,
          modelName: config.modelName,
        })

        if (res.speakers) {
          setSpeakers(res.speakers)
        }

        if (res.turns && res.turns.length > 0) {
          playBlip("speaker")
          setTurns((prev) => {
            const updated = [...prev]
            for (const newTurn of res.turns) {
              updated.push({
                ...newTurn,
                timestamp: new Date().toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                }),
                procTime: res.processing_time,
              })
            }
            return updated
          })
          const lastTurn = res.turns[res.turns.length - 1]
          const langList = (lastTurn.languages || [])
            .map((l) => l.name)
            .join(" + ")
          setStatusMessage(
            `${lastTurn.speaker_label} · ${langList} (${res.processing_time}s)`,
          )
        } else {
          setStatusMessage("No clear speech detected in segment")
        }
      } catch (err) {
        console.error("Transcription error:", err)
        setStatusMessage(`Error: ${err.message}`)
      } finally {
        setIsProcessing(false)
      }
    },
    [config, selectedLangs],
  )

  // Push-to-talk start/stop
  const handlePushToTalkStart = useCallback(async () => {
    if (isRecording) return
    playBlip("ping")
    setStatusMessage("Listening (Push-to-Talk)... Speak in Telugu, Hindi, or English")
    await startRecording({ continuous: false })
  }, [isRecording, startRecording])

  const handlePushToTalkStop = useCallback(async () => {
    if (!isRecording || isContinuous) return
    const audioData = await stopRecording()
    if (audioData && audioData.base64Data) {
      await processAudioChunk(audioData.base64Data)
    } else {
      setStatusMessage("Ready — Hold SPACE to talk, or enable LIVE AUTO-LISTEN")
    }
  }, [isRecording, isContinuous, stopRecording, processAudioChunk])

  // Continuous hands-free mode toggle
  const handleToggleContinuous = useCallback(async () => {
    if (isRecording && isContinuous) {
      const remaining = await stopRecording()
      setStatusMessage("Live Auto-Listen paused")
      if (remaining && remaining.base64Data && remaining.durationSec >= 0.5) {
        await processAudioChunk(remaining.base64Data)
      }
      return
    }
    if (isRecording) {
      await stopRecording()
    }
    playBlip("ping")
    setStatusMessage(
      "🔴 LIVE AUTO-LISTEN ACTIVE — Speak naturally & switch languages anytime",
    )
    await startRecording({
      continuous: true,
      onChunkReady: (chunk) => {
        if (chunk && chunk.base64Data) {
          processAudioChunk(chunk.base64Data)
        }
      },
    })
  }, [isRecording, isContinuous, startRecording, stopRecording, processAudioChunk])

  // Keyboard shortcuts: Hold SPACE or Z for Push-to-Talk, press L for Live Auto-Listen
  useEffect(() => {
    const onKeyDown = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return
      if (editingSpeaker || showEnrollPanel) return

      if ((e.key === " " || e.key.toLowerCase() === "z") && !e.repeat) {
        e.preventDefault()
        if (!isRecording) {
          handlePushToTalkStart()
        }
      } else if (e.key.toLowerCase() === "l" && !e.repeat) {
        e.preventDefault()
        handleToggleContinuous()
      }
    }

    const onKeyUp = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return
      if (editingSpeaker || showEnrollPanel) return

      if (e.key === " " || e.key.toLowerCase() === "z") {
        e.preventDefault()
        if (isRecording && !isContinuous) {
          handlePushToTalkStop()
        }
      }
    }

    window.addEventListener("keydown", onKeyDown)
    window.addEventListener("keyup", onKeyUp)
    return () => {
      window.removeEventListener("keydown", onKeyDown)
      window.removeEventListener("keyup", onKeyUp)
    }
  }, [
    isRecording,
    isContinuous,
    editingSpeaker,
    showEnrollPanel,
    handlePushToTalkStart,
    handlePushToTalkStop,
    handleToggleContinuous,
  ])

  const toggleLanguageChip = (code) => {
    playBlip("language")
    setSelectedLangs((prev) => {
      if (prev.includes(code)) {
        return prev.filter((c) => c !== code)
      }
      return [...prev, code]
    })
  }

  const setPresetMainTriad = () => {
    playBlip("language")
    setSelectedLangs(["te", "en", "hi"])
  }

  const setPresetAllLanguages = () => {
    playBlip("language")
    setSelectedLangs([])
  }

  // Speaker rename handler
  const handleSaveSpeakerRename = async (e) => {
    e.preventDefault()
    if (!editingSpeaker || !renameValue.trim()) {
      setEditingSpeaker(null)
      return
    }
    const newLabel = renameValue.trim()
    try {
      const res = await renameSpeakerAPI(editingSpeaker.id, newLabel)
      if (res.speakers) setSpeakers(res.speakers)
      setTurns((prev) =>
        prev.map((t) =>
          t.speaker_id === editingSpeaker.id
            ? { ...t, speaker_label: newLabel }
            : t,
        ),
      )
      setEditingSpeaker(null)
      setRenameValue("")
    } catch (err) {
      console.error(err)
    }
  }

  // Voiceprint enrollment handlers
  const handleEnrollStart = async () => {
    if (!enrollingName.trim() || isRecording) return
    setIsEnrollingRecord(true)
    playBlip("ping")
    await startRecording({ continuous: false })
  }

  const handleEnrollStop = async () => {
    if (!isEnrollingRecord) return
    setIsEnrollingRecord(false)
    const audioData = await stopRecording()
    if (audioData && audioData.base64Data) {
      try {
        setStatusMessage(`Enrolling voiceprint for ${enrollingName.trim()}...`)
        const res = await enrollSpeakerAPI(
          enrollingName.trim(),
          audioData.base64Data,
        )
        if (res.speakers) setSpeakers(res.speakers)
        setStatusMessage(
          `Enrolled speaker "${enrollingName.trim()}" successfully!`,
        )
        setEnrollingName("")
        setShowEnrollPanel(false)
      } catch (err) {
        setStatusMessage(`Enrollment failed: ${err.message}`)
      }
    }
  }

  const handleResetSpeakers = async () => {
    try {
      await resetSpeakersAPI()
      setSpeakers([])
      setStatusMessage("Speaker voiceprints reset for new conversation")
    } catch (err) {
      console.error(err)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ""
    try {
      setIsProcessing(true)
      setStatusMessage(`Decoding audio file "${file.name}"...`)
      const decoded = await decodeAudioFileToBase64(file)
      if (decoded && decoded.base64Data) {
        await processAudioChunk(decoded.base64Data)
      }
    } catch (err) {
      setStatusMessage(`File error: ${err.message}`)
      setIsProcessing(false)
    }
  }

  const handleCopyTranscript = () => {
    if (turns.length === 0) return
    const formatted = turns
      .map((t) => {
        const langs = (t.languages || []).map((l) => l.name).join(" → ")
        const trans =
          t.english_translation &&
          t.english_translation.toLowerCase() !== t.text.toLowerCase()
            ? `\n   [Meaning: ${t.english_translation}]`
            : ""
        return `[${t.timestamp || ""}] ${t.speaker_label} (${langs}):\n${t.text}${trans}`
      })
      .join("\n\n")
    navigator.clipboard.writeText(formatted)
    setStatusMessage("Transcript copied to clipboard!")
  }

  const handleSpeakTurn = (turn) => {
    if (!turn || !turn.text) return
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause()
      audioPlayerRef.current = null
    }
    const primaryLang = turn.languages?.[0]?.code || "en"
    const supportedMoonshine = ["en", "es", "ar", "ja", "zh", "ko"]
    if (supportedMoonshine.includes(primaryLang)) {
      const chunks = splitTextIntoSpeechChunks(turn.text)
      if (!chunks.length) return
      const url = `/api/tts?text=${encodeURIComponent(chunks[0])}&lang=${encodeURIComponent(primaryLang)}`
      const player = new Audio(url)
      audioPlayerRef.current = player
      player.play().catch(() => {})
    } else if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel()
      const utter = new SpeechSynthesisUtterance(turn.text)
      const langMap = { te: "te-IN", hi: "hi-IN", ta: "ta-IN", kn: "kn-IN" }
      utter.lang = langMap[primaryLang] || primaryLang
      window.speechSynthesis.speak(utter)
    }
  }

  const isMainTriadSelected =
    selectedLangs.length === 3 &&
    selectedLangs.includes("te") &&
    selectedLangs.includes("en") &&
    selectedLangs.includes("hi")

  return (
    <div className="transcriber-shell">
      {/* Top Status & Language Bar */}
      <header className="transcriber-header">
        <div className="header-top-row">
          <div className="brand-block">
            <span className="brand-badge">GEMMA-4 OMNI TRANSCRIBER</span>
            <span className="engine-tag">
              {config.asrEngine === "whisper"
                ? `WHISPER (${(config.whisperModel || "small").toUpperCase()}) + 3D-SPEAKER`
                : config.asrEngine === "hybrid"
                  ? "HYBRID (GEMMA-4 + WHISPER + 3D-SPEAKER)"
                  : "GEMMA-4 AUDIO + 3D-SPEAKER DIARIZER"}
            </span>
          </div>
          <div className="header-actions">
            <button
              className="retro-btn btn-xs"
              onClick={handleCopyTranscript}
              disabled={turns.length === 0}
              title="Copy full transcript"
            >
              COPY
            </button>
            <button
              className="retro-btn btn-xs"
              onClick={() => setTurns([])}
              disabled={turns.length === 0}
              title="Clear transcript"
            >
              CLEAR
            </button>
          </div>
        </div>

        {/* Language Code-Switching Bar */}
        <div className="lang-filter-bar">
          <span className="bar-label">LANGUAGES:</span>
          <button
            className={`lang-chip preset-chip ${isMainTriadSelected ? "active" : ""}`}
            onClick={setPresetMainTriad}
            title="Optimize for Telugu, English, and Hindi code-switching"
          >
            ★ TE + EN + HI
          </button>
          <button
            className={`lang-chip preset-chip ${selectedLangs.length === 0 ? "active" : ""}`}
            onClick={setPresetAllLanguages}
            title="Auto-detect across all 99+ languages"
          >
            🌐 ALL 99+
          </button>
          <span className="bar-divider">|</span>
          <div className="lang-chips-scroll">
            {QUICK_LANGUAGES.map((l) => {
              const active = selectedLangs.includes(l.code)
              return (
                <button
                  key={l.code}
                  className={`lang-chip ${active ? "active" : ""}`}
                  onClick={() => toggleLanguageChip(l.code)}
                  title={`${l.name} (${l.code})`}
                >
                  {l.code.toUpperCase()} · {l.native}
                </button>
              )
            })}
          </div>
        </div>

        {/* Speaker Diarization Roster Bar */}
        <div className="speaker-roster-bar">
          <span className="bar-label">SPEAKERS ({speakers.length}):</span>
          <div className="speaker-pills-wrap">
            {speakers.length === 0 ? (
              <span className="speaker-empty-hint">
                Auto-detects & separates voices as people speak...
              </span>
            ) : (
              speakers.map((spk) => (
                <button
                  key={spk.id}
                  className="speaker-pill"
                  onClick={() => {
                    setEditingSpeaker(spk)
                    setRenameValue(spk.label)
                  }}
                  title="Click to rename this speaker"
                >
                  <span className="speaker-dot">●</span>
                  <span className="speaker-pill-name">{spk.label}</span>
                  <span className="speaker-pill-meta">
                    {spk.utterance_count}x · {spk.total_seconds}s ✎
                  </span>
                </button>
              ))
            )}
          </div>
          <div className="speaker-bar-actions">
            <button
              className="retro-btn btn-xs"
              onClick={() => setShowEnrollPanel((p) => !p)}
              title="Enroll a speaker's voiceprint"
            >
              + ENROLL VOICE
            </button>
            {speakers.length > 0 && (
              <button
                className="retro-btn btn-xs"
                onClick={handleResetSpeakers}
                title="Reset speaker voiceprints"
              >
                RESET IDs
              </button>
            )}
          </div>
        </div>

        {/* Inline Rename Speaker Bar */}
        {editingSpeaker && (
          <form className="inline-modal-bar" onSubmit={handleSaveSpeakerRename}>
            <span>RENAME {editingSpeaker.label.toUpperCase()}:</span>
            <input
              type="text"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              placeholder="Enter speaker name (e.g. Sandeep)"
              autoFocus
            />
            <button type="submit" className="retro-btn btn-xs">
              SAVE
            </button>
            <button
              type="button"
              className="retro-btn btn-xs"
              onClick={() => setEditingSpeaker(null)}
            >
              CANCEL
            </button>
          </form>
        )}

        {/* Inline Voiceprint Enrollment Bar */}
        {showEnrollPanel && (
          <div className="inline-modal-bar">
            <span>ENROLL SPEAKER:</span>
            <input
              type="text"
              value={enrollingName}
              onChange={(e) => setEnrollingName(e.target.value)}
              placeholder="Speaker Name (e.g. Sandeep)"
            />
            <button
              type="button"
              className={`retro-btn btn-xs ${isEnrollingRecord ? "recording-active" : ""}`}
              disabled={!enrollingName.trim()}
              onMouseDown={handleEnrollStart}
              onMouseUp={handleEnrollStop}
              onTouchStart={handleEnrollStart}
              onTouchEnd={handleEnrollStop}
            >
              {isEnrollingRecord
                ? "🔴 RECORDING VOICE... RELEASE TO SAVE"
                : "HOLD TO RECORD 3s VOICE SAMPLE"}
            </button>
            <button
              type="button"
              className="retro-btn btn-xs"
              onClick={() => setShowEnrollPanel(false)}
            >
              ✕
            </button>
          </div>
        )}
      </header>

      {/* Main Diarized & Code-Switched Conversation Feed */}
      <main className="transcript-feed">
        {micError && (
          <div className="error-banner">
            ⚠️ Microphone Error: {micError}
          </div>
        )}

        {turns.length === 0 && !isProcessing ? (
          <div className="empty-feed-state">
            <div className="empty-title">
              MULTILINGUAL + MULTI-SPEAKER VOICE TRANSCRIBER
            </div>
            <div className="empty-subtitle">
              Speak naturally and switch between{" "}
              <strong>Telugu (తెలుగు)</strong>, <strong>English</strong>, and{" "}
              <strong>Hindi (हिन्दी)</strong> mid-conversation.
            </div>
            <div className="empty-instructions">
              <div>
                <kbd>HOLD SPACE</kbd> or <kbd>HOLD TALK</kbd> — Push-to-Talk
                recording
              </div>
              <div>
                <kbd>L</kbd> or <kbd>LIVE AUTO-LISTEN</kbd> — Hands-free
                continuous conversation mode
              </div>
              <div>
                Automatic <strong>Speaker Diarization</strong> identifies who is
                speaking & tracks voiceprints across language switches.
              </div>
            </div>
          </div>
        ) : (
          <div className="turns-list">
            {turns.map((turn, idx) => {
              const isEvenSpeaker = (turn.speaker_index || 1) % 2 === 0
              const hasMultipleClauses =
                turn.clauses && turn.clauses.length > 1
              const showTranslation =
                config.showEnglishMeaning !== false &&
                turn.english_translation &&
                turn.english_translation.trim() !== "" &&
                turn.english_translation.trim().toLowerCase() !==
                  turn.text.trim().toLowerCase()

              return (
                <article
                  key={turn.id || idx}
                  className={`turn-card ${isEvenSpeaker ? "turn-right" : "turn-left"} ${turn.language_switched ? "code-switched" : ""}`}
                >
                  <div className="turn-card-header">
                    <div className="turn-speaker-group">
                      <span
                        className="turn-speaker-badge"
                        onClick={() => {
                          setEditingSpeaker({
                            id: turn.speaker_id,
                            label: turn.speaker_label,
                          })
                          setRenameValue(turn.speaker_label)
                        }}
                        title="Click to rename speaker"
                      >
                        👤 {turn.speaker_label}
                      </span>
                      {turn.speaker_confidence && (
                        <span className="turn-conf-badge">
                          {Math.round(turn.speaker_confidence * 100)}% match
                        </span>
                      )}
                    </div>

                    <div className="turn-lang-group">
                      {turn.language_switched && (
                        <span className="code-switch-pill">
                          ⚡ CODE-SWITCH
                        </span>
                      )}
                      {(turn.languages || []).map((l, i) => (
                        <span key={l.code + i} className="turn-lang-pill">
                          {l.name.toUpperCase()}
                        </span>
                      ))}
                      <button
                        className="turn-speak-btn"
                        onClick={() => handleSpeakTurn(turn)}
                        title="Listen to utterance"
                      >
                        🔊
                      </button>
                    </div>
                  </div>

                  {/* Clause-level breakdown when speaker switches languages mid-turn */}
                  {hasMultipleClauses ? (
                    <div className="turn-clauses-flow">
                      {turn.clauses.map((c, cIdx) => (
                        <span key={cIdx} className="clause-segment">
                          <span className="clause-lang-tag">
                            {(c.lang || "en").toUpperCase()}
                          </span>
                          <span className="clause-text">{c.text}</span>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className="turn-main-text">{turn.text}</div>
                  )}

                  {showTranslation && (
                    <div className="turn-translation-row">
                      <span className="translation-prefix">↳ EN MEANING:</span>{" "}
                      {turn.english_translation}
                    </div>
                  )}

                  <div className="turn-footer-meta">
                    <span>{turn.timestamp}</span>
                    <span>
                      Duration:{" "}
                      {Math.max(0.1, (turn.end - turn.start).toFixed(1))}s
                    </span>
                    {turn.procTime && <span> Inference: {turn.procTime}s</span>}
                  </div>
                </article>
              )
            })}

            {isProcessing && (
              <div className="processing-indicator">
                <span className="pulse-dot">■</span> DIARIZING SPEAKER &
                TRANSCRIBING MULTILINGUAL SPEECH...
              </div>
            )}
            <div ref={feedEndRef} />
          </div>
        )}
      </main>

      {/* Frequency Visualizer Bar */}
      <Visualizer
        activePerson={1}
        isRecording={isRecording}
        analyser={analyser}
        barsCount={parseInt(config.visualizerBars || 32, 10)}
      />

      {/* Bottom Control Deck */}
      <footer className="transcriber-controls">
        <div className="status-ticker">{statusMessage}</div>
        <div className="control-buttons-row">
          <button
            className={`control-btn ptt-btn ${isRecording && !isContinuous ? "recording" : ""}`}
            onMouseDown={handlePushToTalkStart}
            onMouseUp={handlePushToTalkStop}
            onTouchStart={(e) => {
              e.preventDefault()
              handlePushToTalkStart()
            }}
            onTouchEnd={(e) => {
              e.preventDefault()
              handlePushToTalkStop()
            }}
            disabled={isContinuous}
          >
            {isRecording && !isContinuous
              ? "● RECORDING... RELEASE TO TRANSCRIBE"
              : "🎙️ HOLD TO TALK [SPACE]"}
          </button>

          <button
            className={`control-btn continuous-btn ${isContinuous ? "recording" : ""}`}
            onClick={handleToggleContinuous}
          >
            {isContinuous
              ? "⏹ STOP LIVE AUTO-LISTEN [L]"
              : "♾️ LIVE AUTO-LISTEN [L]"}
          </button>

          <button
            className="control-btn upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isRecording || isProcessing}
            title="Upload an audio file (.wav, .mp3, .m4a) to diarize & transcribe"
          >
            📁 FILE
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            style={{ display: "none" }}
            onChange={handleFileUpload}
          />
        </div>
      </footer>
    </div>
  )
}
