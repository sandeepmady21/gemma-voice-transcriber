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
import { playBlip } from "./utils/audio-blip"
import {
  transcribeConversationChunk,
  fetchSpeakers,
  renameSpeakerAPI,
  enrollSpeakerAPI,
  resetSpeakersAPI,
  splitTextIntoSpeechChunks,
} from "./utils/api"

const LANG_COLORS = {
  te: { bg: "rgba(16, 185, 129, 0.16)", border: "#10b981", text: "#34d399" },
  en: { bg: "rgba(59, 130, 246, 0.16)", border: "#3b82f6", text: "#60a5fa" },
  hi: { bg: "rgba(245, 158, 11, 0.16)", border: "#f59e0b", text: "#fbbf24" },
}

function getLangStyle(code) {
  return (
    LANG_COLORS[code] || {
      bg: "rgba(148, 163, 184, 0.16)",
      border: "#64748b",
      text: "#cbd5e1",
    }
  )
}

export default function TranscriberApp({ config, onOpenSettings }) {
  const {
    isRecording,
    isContinuous,
    startRecording,
    stopRecording,
    decodeAudioFileToBase64,
    analyser,
    micError,
  } = useAudioRecorder()

  const [turns, setTurns] = useState([])
  const [speakers, setSpeakers] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [statusMessage, setStatusMessage] = useState(
    "Ready — Click Record or Auto-Listen to transcribe Telugu, English & Hindi",
  )

  // "roman" = English script for Telugu, Hindi & English (default as requested)
  // "native" = Native script (తెలుగు / हिन्दी / English)
  const [scriptMode, setScriptMode] = useState(
    () => localStorage.getItem("transcriber_script_mode") || "roman",
  )

  // "feed" = Diarized Speaker Cards, "document" = Full Text Document
  const [viewMode, setViewMode] = useState("feed")

  // Active language filter preset: ["te", "en", "hi"] by default
  const [selectedLangs, setSelectedLangs] = useState(["te", "en", "hi"])

  // Recording elapsed timer
  const [recordingSeconds, setRecordingSeconds] = useState(0)

  // Speaker rename & voice enrollment state
  const [editingSpeakerId, setEditingSpeakerId] = useState(null)
  const [renameValue, setRenameValue] = useState("")
  const [enrollingName, setEnrollingName] = useState("")
  const [isEnrollingRecord, setIsEnrollingRecord] = useState(false)
  const [copiedId, setCopiedId] = useState(null)

  const feedEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const audioPlayerRef = useRef(null)
  const chunkQueueRef = useRef([])
  const isDrainingQueueRef = useRef(false)

  useEffect(() => {
    localStorage.setItem("transcriber_script_mode", scriptMode)
  }, [scriptMode])

  useEffect(() => {
    fetchSpeakers()
      .then((data) => {
        if (data.speakers) setSpeakers(data.speakers)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (feedEndRef.current && viewMode === "feed") {
      feedEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [turns, isProcessing, viewMode])

  useEffect(() => {
    if (!isRecording) {
      setRecordingSeconds(0)
      return
    }
    const timer = setInterval(() => {
      setRecordingSeconds((s) => s + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [isRecording])

  const formatTimer = (secs) => {
    const m = String(Math.floor(secs / 60)).padStart(2, "0")
    const s = String(secs % 60).padStart(2, "0")
    return `${m}:${s}`
  }

  const processSingleChunk = useCallback(
    async (base64Data) => {
      if (!base64Data) return
      setIsProcessing(true)
      setStatusMessage("Transcribing speech & matching speaker voiceprint...")

      try {
        const res = await transcribeConversationChunk(base64Data, {
          engine: config.asrEngine || "gemma4-audio",
          whisperModel: config.whisperModel || "small",
          allowedLanguages: selectedLangs,
          speakerThreshold: config.speakerThreshold ?? 0.52,
          endpointUrl: config.endpointUrl,
          modelName: config.modelName,
          scriptMode,
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
            `✓ ${lastTurn.speaker_label} · ${langList} (${res.processing_time}s)`,
          )
        } else {
          setStatusMessage(
            isContinuous
              ? "Auto-Listen active — waiting for speech..."
              : "No clear speech detected — speak closer to the mic",
          )
        }
      } catch (err) {
        console.error("Transcription error:", err)
        setStatusMessage(`Error: ${err.message}`)
      } finally {
        setIsProcessing(false)
      }
    },
    [config, selectedLangs, scriptMode, isContinuous],
  )

  const drainQueue = useCallback(async () => {
    if (isDrainingQueueRef.current) return
    isDrainingQueueRef.current = true
    try {
      while (chunkQueueRef.current.length > 0) {
        const nextBase64 = chunkQueueRef.current.shift()
        await processSingleChunk(nextBase64)
      }
    } finally {
      isDrainingQueueRef.current = false
    }
  }, [processSingleChunk])

  const enqueueAudioChunk = useCallback(
    (base64Data) => {
      if (!base64Data) return
      chunkQueueRef.current.push(base64Data)
      drainQueue()
    },
    [drainQueue],
  )

  // Click-to-Record / Click-to-Stop toggle
  const handleToggleRecord = useCallback(async () => {
    if (isContinuous) {
      const leftover = await stopRecording()
      if (leftover && leftover.base64Data && leftover.durationSec >= 0.4) {
        enqueueAudioChunk(leftover.base64Data)
      } else {
        setStatusMessage("Stopped Auto-Listen.")
      }
      return
    }

    if (isRecording) {
      const audioData = await stopRecording()
      if (audioData && audioData.base64Data) {
        enqueueAudioChunk(audioData.base64Data)
      } else {
        setStatusMessage("Ready — Click Record to start speaking")
      }
    } else {
      playBlip("ping")
      setStatusMessage(
        "Recording... Speak in Telugu, Hindi, or English, then click Stop & Transcribe",
      )
      await startRecording({ continuous: false })
    }
  }, [
    isRecording,
    isContinuous,
    startRecording,
    stopRecording,
    enqueueAudioChunk,
  ])

  // Continuous hands-free mode toggle
  const handleToggleContinuous = useCallback(async () => {
    if (isRecording && isContinuous) {
      const leftover = await stopRecording()
      if (leftover && leftover.base64Data && leftover.durationSec >= 0.45) {
        enqueueAudioChunk(leftover.base64Data)
      }
      setStatusMessage("Auto-Listen paused.")
    } else {
      if (isRecording) {
        await stopRecording()
      }
      playBlip("ping")
      setStatusMessage(
        "Auto-Listen ON — Speak naturally; pauses automatically transcribe",
      )
      await startRecording({
        continuous: true,
        onChunkReady: (chunkObj) => {
          if (chunkObj && chunkObj.base64Data) {
            enqueueAudioChunk(chunkObj.base64Data)
          }
        },
      })
    }
  }, [
    isRecording,
    isContinuous,
    startRecording,
    stopRecording,
    enqueueAudioChunk,
  ])

  // Spacebar shortcut to toggle recording
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (
        e.target.tagName === "INPUT" ||
        e.target.tagName === "TEXTAREA" ||
        e.target.tagName === "SELECT" ||
        editingSpeakerId
      ) {
        return
      }
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault()
        handleToggleRecord()
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [editingSpeakerId, handleToggleRecord])

  const handleSaveRename = async (spkId) => {
    if (!renameValue.trim()) {
      setEditingSpeakerId(null)
      return
    }
    const newLabel = renameValue.trim()
    try {
      const res = await renameSpeakerAPI(spkId, newLabel)
      if (res.speakers) setSpeakers(res.speakers)
      setTurns((prev) =>
        prev.map((t) =>
          t.speaker_id === spkId ? { ...t, speaker_label: newLabel } : t,
        ),
      )
      setEditingSpeakerId(null)
      setRenameValue("")
    } catch (err) {
      console.error(err)
    }
  }

  const handleEnrollToggle = async () => {
    if (!enrollingName.trim()) return
    if (!isEnrollingRecord) {
      setIsEnrollingRecord(true)
      playBlip("ping")
      await startRecording({ continuous: false })
    } else {
      setIsEnrollingRecord(false)
      const audioData = await stopRecording()
      if (audioData && audioData.base64Data) {
        try {
          setStatusMessage(
            `Enrolling voiceprint for ${enrollingName.trim()}...`,
          )
          const res = await enrollSpeakerAPI(
            enrollingName.trim(),
            audioData.base64Data,
          )
          if (res.speakers) setSpeakers(res.speakers)
          setStatusMessage(`✓ Enrolled speaker "${enrollingName.trim()}"!`)
          setEnrollingName("")
        } catch (err) {
          setStatusMessage(`Enrollment failed: ${err.message}`)
        }
      }
    }
  }

  const handleResetSpeakers = async () => {
    try {
      await resetSpeakersAPI()
      setSpeakers([])
      setStatusMessage("Speaker voiceprints reset for new session.")
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
        enqueueAudioChunk(decoded.base64Data)
      }
    } catch (err) {
      setStatusMessage(`File error: ${err.message}`)
      setIsProcessing(false)
    }
  }

  const buildFullText = useCallback(() => {
    if (turns.length === 0) return ""
    return turns
      .map((t) => {
        const langs = (t.languages || []).map((l) => l.name).join(" + ")
        const meaning =
          t.english_translation &&
          t.english_translation.toLowerCase() !== t.text.toLowerCase()
            ? `\n  [English Meaning: ${t.english_translation}]`
            : ""
        return `[${t.timestamp || ""}] ${t.speaker_label} (${langs}): ${t.text}${meaning}`
      })
      .join("\n\n")
  }, [turns])

  const handleCopyAll = () => {
    if (turns.length === 0) return
    navigator.clipboard.writeText(buildFullText())
    setCopiedId("all")
    setTimeout(() => setCopiedId(null), 1800)
    setStatusMessage("✓ Copied full transcript to clipboard!")
  }

  const handleCopyTurn = (turn) => {
    navigator.clipboard.writeText(turn.text)
    setCopiedId(turn.id)
    setTimeout(() => setCopiedId(null), 1500)
  }

  const handleExportTxt = () => {
    if (turns.length === 0) return
    const blob = new Blob([buildFullText()], {
      type: "text/plain;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `transcript-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.txt`
    a.click()
    URL.revokeObjectURL(url)
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

  const isTriadActive =
    selectedLangs.length === 3 &&
    selectedLangs.includes("te") &&
    selectedLangs.includes("en") &&
    selectedLangs.includes("hi")

  return (
    <div className="studio-shell">
      {/* Top Studio Header */}
      <header className="studio-header">
        <div className="studio-brand">
          <span className="studio-logo-dot" />
          <span className="studio-title">GEMMA-4 VOICE STUDIO</span>
          <span className="studio-subtitle">
            Telugu · English · Hindi · Multi-Speaker
          </span>
        </div>

        <div className="studio-toolbar-center">
          {/* Script Output Mode: English Script (Romanized) vs Native Script */}
          <div
            className="script-mode-group"
            title="Write Telugu & Hindi in English alphabet (Romanized) or Native script"
          >
            <button
              className={`mode-pill ${scriptMode === "roman" ? "active" : ""}`}
              onClick={() => setScriptMode("roman")}
            >
              Aa English Script (TE/HI/EN)
            </button>
            <button
              className={`mode-pill ${scriptMode === "native" ? "active" : ""}`}
              onClick={() => setScriptMode("native")}
            >
              ఆ/अ Native Script
            </button>
          </div>

          {/* View Mode: Speaker Feed vs Full Text Document */}
          <div className="view-mode-group">
            <button
              className={`mode-pill ${viewMode === "feed" ? "active" : ""}`}
              onClick={() => setViewMode("feed")}
            >
              💬 Speaker Feed
            </button>
            <button
              className={`mode-pill ${viewMode === "document" ? "active" : ""}`}
              onClick={() => setViewMode("document")}
            >
              📄 Full Text
            </button>
          </div>
        </div>

        <div className="studio-actions">
          <button
            className="studio-action-btn"
            onClick={handleCopyAll}
            disabled={turns.length === 0}
          >
            {copiedId === "all" ? "✓ Copied!" : "📋 Copy All"}
          </button>
          <button
            className="studio-action-btn"
            onClick={handleExportTxt}
            disabled={turns.length === 0}
          >
            ⬇ Export .txt
          </button>
          <button
            className="studio-action-btn danger"
            onClick={() => setTurns([])}
            disabled={turns.length === 0}
          >
            Clear
          </button>
          {onOpenSettings && (
            <button
              className="studio-action-btn"
              onClick={onOpenSettings}
              title="Settings"
            >
              ⚙️
            </button>
          )}
        </div>
      </header>

      {/* Speaker & Language Roster Bar */}
      <div className="studio-roster-bar">
        <div className="roster-left">
          <span className="roster-label">LANGUAGES:</span>
          <button
            className={`mode-pill ${isTriadActive ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: "12px" }}
            onClick={() => setSelectedLangs(["te", "en", "hi"])}
          >
            ★ Auto (TE + EN + HI)
          </button>
          <button
            className={`mode-pill ${selectedLangs.length === 1 && selectedLangs[0] === "te" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: "12px" }}
            onClick={() => setSelectedLangs(["te"])}
          >
            Telugu
          </button>
          <button
            className={`mode-pill ${selectedLangs.length === 1 && selectedLangs[0] === "en" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: "12px" }}
            onClick={() => setSelectedLangs(["en"])}
          >
            English
          </button>
          <button
            className={`mode-pill ${selectedLangs.length === 1 && selectedLangs[0] === "hi" ? "active" : ""}`}
            style={{ padding: "4px 10px", fontSize: "12px" }}
            onClick={() => setSelectedLangs(["hi"])}
          >
            Hindi
          </button>

          <span className="roster-label" style={{ marginLeft: "10px" }}>
            SPEAKERS ({speakers.length}):
          </span>
          {speakers.length === 0 ? (
            <span className="roster-hint">
              Auto-detects voices as you speak (click badge to rename)
            </span>
          ) : (
            speakers.map((spk) =>
              editingSpeakerId === spk.id ? (
                <span key={spk.id} className="speaker-rename-inline">
                  <input
                    type="text"
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveRename(spk.id)
                      if (e.key === "Escape") setEditingSpeakerId(null)
                    }}
                    placeholder="Speaker name"
                    autoFocus
                  />
                  <button onClick={() => handleSaveRename(spk.id)}>Save</button>
                </span>
              ) : (
                <button
                  key={spk.id}
                  className="speaker-pill"
                  style={{ borderColor: spk.color }}
                  onClick={() => {
                    setEditingSpeakerId(spk.id)
                    setRenameValue(spk.label)
                  }}
                  title="Click to rename speaker"
                >
                  <span
                    className="speaker-dot"
                    style={{ background: spk.color }}
                  />
                  <span>{spk.label}</span>
                  <span className="speaker-edit-icon">✎</span>
                </button>
              ),
            )
          )}
        </div>

        <div className="roster-right">
          <div className="enroll-inline">
            <input
              type="text"
              placeholder="Enroll name (e.g. Sandeep)"
              value={enrollingName}
              onChange={(e) => setEnrollingName(e.target.value)}
            />
            {enrollingName.trim() && (
              <button
                className={`enroll-btn ${isEnrollingRecord ? "recording" : ""}`}
                onClick={handleEnrollToggle}
              >
                {isEnrollingRecord
                  ? "⏹ Stop & Save Voice"
                  : "🎙️ Record 3s Voice"}
              </button>
            )}
          </div>
          {speakers.length > 0 && (
            <button className="reset-voices-btn" onClick={handleResetSpeakers}>
              Reset Voices
            </button>
          )}
        </div>
      </div>

      {/* Main Transcript Area */}
      <main className="studio-workspace">
        {micError && (
          <div
            style={{
              background: "rgba(239, 68, 68, 0.18)",
              border: "1px solid #ef4444",
              color: "#fca5a5",
              padding: "10px 16px",
              borderRadius: "10px",
              marginBottom: "14px",
              width: "100%",
              maxWidth: "880px",
            }}
          >
            ⚠️ Microphone Error: {micError}
          </div>
        )}

        {turns.length === 0 && !isProcessing ? (
          <div className="studio-empty-state">
            <div className="empty-mic-circle">🎙️</div>
            <h2>Ready to Transcribe Telugu, English & Hindi</h2>
            <p>
              Click <strong>Record</strong> below (or press{" "}
              <strong>Space</strong>) and speak naturally. Switch between{" "}
              <strong>Telugu</strong>, <strong>English</strong>, and{" "}
              <strong>Hindi</strong> mid-conversation —{" "}
              <strong>Gemma 4 Audio</strong> + <strong>3D-Speaker</strong>{" "}
              automatically identifies who is speaking and transcribes into{" "}
              <strong>
                {scriptMode === "roman"
                  ? "English Script (Romanized TE/HI/EN)"
                  : "Native Script (తెలుగు / हिन्दी / English)"}
              </strong>
              .
            </p>
            <div className="empty-tips">
              <span className="tip-badge">
                ✓ English Script for Telugu & Hindi
              </span>
              <span className="tip-badge">✓ Multi-Speaker Voiceprint ID</span>
              <span className="tip-badge">✓ Mid-Sentence Code-Switching</span>
            </div>
          </div>
        ) : viewMode === "document" ? (
          <div className="studio-document-container">
            <div className="document-header">
              <span>Full Conversation Transcript ({turns.length} turns)</span>
              <button className="doc-copy-btn" onClick={handleCopyAll}>
                {copiedId === "all" ? "✓ Copied!" : "📋 Copy Full Text"}
              </button>
            </div>
            <div className="document-body">
              {turns.map((turn, idx) => (
                <div key={turn.id || idx} className="document-paragraph">
                  <span
                    className="doc-speaker"
                    style={{ color: turn.speaker_color || "#f59e0b" }}
                  >
                    {turn.speaker_label}
                  </span>
                  <span className="doc-lang-tag">
                    [{(turn.languages || []).map((l) => l.name).join(" + ")}]
                  </span>
                  <span className="doc-text">{turn.text}</span>
                  {config.showEnglishMeaning !== false &&
                    turn.english_translation &&
                    turn.english_translation.toLowerCase() !==
                      turn.text.toLowerCase() && (
                      <span className="doc-meaning">
                        {" "}
                        — ({turn.english_translation})
                      </span>
                    )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="studio-feed-list">
            {turns.map((turn, idx) => {
              const spkColor = turn.speaker_color || "#f59e0b"
              const hasMultiClauses = turn.clauses && turn.clauses.length > 1
              const showMeaning =
                config.showEnglishMeaning !== false &&
                turn.english_translation &&
                turn.english_translation.trim() !== "" &&
                turn.english_translation.trim().toLowerCase() !==
                  turn.text.trim().toLowerCase()

              return (
                <div
                  key={turn.id || idx}
                  className="studio-turn-card"
                  style={{ borderLeftColor: spkColor }}
                >
                  <div className="turn-card-header">
                    <span
                      className="turn-speaker-tag"
                      style={{
                        background: `${spkColor}24`,
                        color: spkColor,
                        border: `1px solid ${spkColor}66`,
                      }}
                      onClick={() => {
                        setEditingSpeakerId(turn.speaker_id)
                        setRenameValue(turn.speaker_label)
                      }}
                      title="Click to rename speaker"
                    >
                      👤 {turn.speaker_label} ✎
                    </span>

                    <div className="turn-meta-badges">
                      {turn.language_switched && (
                        <span className="turn-switch-pill">
                          ⚡ Switched Languages
                        </span>
                      )}
                      {(turn.languages || []).map((l, i) => {
                        const st = getLangStyle(l.code)
                        return (
                          <span
                            key={l.code + i}
                            className="turn-lang-pill"
                            style={{
                              background: st.bg,
                              color: st.text,
                              border: `1px solid ${st.border}`,
                            }}
                          >
                            {l.name}
                          </span>
                        )
                      })}
                      <span className="turn-time">{turn.timestamp}</span>
                      <button
                        className="turn-copy-btn"
                        onClick={() => handleCopyTurn(turn)}
                        title="Copy text"
                      >
                        {copiedId === turn.id ? "✓" : "📋"}
                      </button>
                      <button
                        className="turn-copy-btn"
                        onClick={() => handleSpeakTurn(turn)}
                        title="Listen"
                      >
                        🔊
                      </button>
                    </div>
                  </div>

                  <div className="turn-card-text">
                    {hasMultiClauses ? (
                      turn.clauses.map((c, cIdx) => {
                        const st = getLangStyle(c.lang)
                        return (
                          <span
                            key={cIdx}
                            className="clause-inline"
                            style={{ borderBottomColor: st.border }}
                            title={c.lang_name || c.lang}
                          >
                            {c.text}
                            {cIdx < turn.clauses.length - 1 ? " " : ""}
                          </span>
                        )
                      })
                    ) : (
                      <span>{turn.text}</span>
                    )}
                  </div>

                  {showMeaning && (
                    <div className="turn-card-meaning">
                      <span className="meaning-prefix">English Meaning:</span>{" "}
                      {turn.english_translation}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {isProcessing && (
          <div className="studio-processing-card">
            <span className="processing-spinner" />
            <span>
              Transcribing speech & matching speaker voiceprint (Gemma 4 +
              3D-Speaker)...
            </span>
          </div>
        )}

        <div ref={feedEndRef} />
      </main>

      {/* Bottom Recording Dock */}
      <footer className="studio-footer">
        <div className="studio-waveform-row">
          <div className="studio-waveform-wrap">
            <Visualizer
              activePerson={1}
              isRecording={isRecording}
              analyser={analyser}
              barsCount={28}
            />
          </div>
          <div
            className={`studio-status-text ${isRecording ? "live" : ""}`}
          >
            {statusMessage}
          </div>
        </div>

        <div className="studio-controls-dock">
          {/* Primary Click-to-Record / Click-to-Stop Button */}
          <button
            className={`dock-btn primary-record ${
              isRecording && !isContinuous ? "recording" : ""
            }`}
            onClick={handleToggleRecord}
          >
            {isRecording && !isContinuous ? (
              <>
                <span className="rec-stop-square" />
                <span>
                  Stop & Transcribe ({formatTimer(recordingSeconds)})
                </span>
              </>
            ) : (
              <>
                <span className="rec-dot" />
                <span>Click to Record (Space)</span>
              </>
            )}
          </button>

          {/* Hands-Free Auto-Listen Toggle */}
          <button
            className={`dock-btn auto-listen ${isContinuous ? "active" : ""}`}
            onClick={handleToggleContinuous}
          >
            {isContinuous
              ? `🟢 Auto-Listening (${formatTimer(recordingSeconds)}) — Click to Stop`
              : "🔄 Auto-Listen (Hands-Free)"}
          </button>

          {/* Upload Audio File */}
          <button
            className="dock-btn secondary"
            onClick={() => fileInputRef.current?.click()}
            disabled={isRecording || isProcessing}
          >
            📁 Upload Audio
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
