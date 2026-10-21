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
import { useAudioRecorder } from "./hooks/useAudioRecorder"
import { playBlip } from "./utils/audio-blip"
import {
  transcribeConversationChunk,
  fetchSpeakers,
  renameSpeakerAPI,
  enrollSpeakerAPI,
  resetSpeakersAPI,
  splitTextIntoSpeechChunks,
  fetchGeminiKeyStatus,
  saveAndTestGeminiKey,
} from "./utils/api"

const LANG_META = {
  te: {
    name: "Telugu",
    bg: "rgba(16, 185, 129, 0.12)",
    border: "rgba(16, 185, 129, 0.35)",
    text: "#34d399",
    dot: "#10b981",
  },
  en: {
    name: "English",
    bg: "rgba(99, 102, 241, 0.12)",
    border: "rgba(99, 102, 241, 0.35)",
    text: "#818cf8",
    dot: "#6366f1",
  },
  hi: {
    name: "Hindi",
    bg: "rgba(245, 158, 11, 0.12)",
    border: "rgba(245, 158, 11, 0.35)",
    text: "#fbbf24",
    dot: "#f59e0b",
  },
}

function getLangStyle(code) {
  return (
    LANG_META[code] || {
      name: code?.toUpperCase() || "Speech",
      bg: "rgba(148, 163, 184, 0.12)",
      border: "rgba(148, 163, 184, 0.3)",
      text: "#cbd5e1",
      dot: "#94a3b8",
    }
  )
}

function getInitials(label = "") {
  const parts = label.trim().split(/\s+/)
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase()
  }
  return label.slice(0, 2).toUpperCase() || "S1"
}

/**
 * Clean, modern audio frequency visualizer for the floating studio dock.
 */
function StudioWaveform({ isRecording, analyser }) {
  const canvasRef = useRef(null)
  const animRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    const bars = 28

    const renderIdle = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const gap = 3
      const barW = (canvas.width - gap * (bars - 1)) / bars
      const cy = canvas.height / 2
      ctx.fillStyle = "rgba(148, 163, 184, 0.25)"
      for (let i = 0; i < bars; i++) {
        const x = i * (barW + gap)
        ctx.beginPath()
        ctx.roundRect(x, cy - 1.5, barW, 3, 1.5)
        ctx.fill()
      }
    }

    if (!isRecording || !analyser) {
      if (animRef.current) cancelAnimationFrame(animRef.current)
      renderIdle()
      return
    }

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const renderLive = () => {
      animRef.current = requestAnimationFrame(renderLive)
      analyser.getByteFrequencyData(dataArray)
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const gap = 3
      const barW = (canvas.width - gap * (bars - 1)) / bars
      const cy = canvas.height / 2
      const maxBin = Math.floor(bufferLength * 0.7)
      const binsPerBar = Math.max(1, Math.floor(maxBin / bars))

      for (let i = 0; i < bars; i++) {
        let sum = 0
        for (let j = 0; j < binsPerBar; j++) {
          sum += dataArray[i * binsPerBar + j] || 0
        }
        const avg = sum / binsPerBar
        const norm = Math.min(1, avg / 210)
        const h = Math.max(3, norm * (canvas.height - 4))
        const x = i * (barW + gap)
        const y = cy - h / 2

        ctx.fillStyle = norm > 0.15 ? "#6366f1" : "rgba(148, 163, 184, 0.35)"
        ctx.beginPath()
        ctx.roundRect(x, y, barW, h, 2)
        ctx.fill()
      }
    }

    renderLive()
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [isRecording, analyser])

  return (
    <canvas
      ref={canvasRef}
      width={140}
      height={28}
      className="pro-waveform-canvas"
    />
  )
}

export default function TranscriberApp({ config, setConfig, onOpenSettings }) {
  const {
    isRecording,
    isContinuous,
    livePreviewText,
    setLivePreviewText,
    startRecording,
    stopRecording,
    decodeAudioFileToBase64,
    analyser,
    micError,
  } = useAudioRecorder()

  const [turns, setTurns] = useState(() => {
    try {
      const saved = localStorage.getItem("transcriber_saved_turns_v1")
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed)) return parsed
      }
    } catch (e) {
      // Ignore storage parse error
    }
    return []
  })
  const [speakers, setSpeakers] = useState([])
  const [inFlightCount, setInFlightCount] = useState(0)
  const [statusMessage, setStatusMessage] = useState("Ready to transcribe")

  const [scriptMode, setScriptMode] = useState(
    () => localStorage.getItem("transcriber_script_mode") || "roman",
  )
  const [viewMode, setViewMode] = useState("feed")
  const [selectedLangs, setSelectedLangs] = useState(["te", "en", "hi"])
  const [recordingSeconds, setRecordingSeconds] = useState(0)

  // Gemini API Key state
  const [hasServerKey, setHasServerKey] = useState(false)
  const [maskedServerKey, setMaskedServerKey] = useState("")
  const [showKeyEditor, setShowKeyEditor] = useState(false)
  const [keyInput, setKeyInput] = useState(() => config.geminiApiKey || "")
  const [isTestingKey, setIsTestingKey] = useState(false)
  const [keyFeedback, setKeyFeedback] = useState(null)

  // Speaker rename & voice enrollment state
  const [editingSpeakerId, setEditingSpeakerId] = useState(null)
  const [renameValue, setRenameValue] = useState("")
  const [showEnrollInput, setShowEnrollInput] = useState(false)
  const [enrollingName, setEnrollingName] = useState("")
  const [isEnrollingRecord, setIsEnrollingRecord] = useState(false)
  const [copiedId, setCopiedId] = useState(null)

  const feedEndRef = useRef(null)
  const prevTurnsLenRef = useRef(turns.length)
  const fileInputRef = useRef(null)
  const audioPlayerRef = useRef(null)

  const activeEngine = config.asrEngine || "gemini-cloud"
  const isGeminiReady = Boolean(
    hasServerKey || (config.geminiApiKey && config.geminiApiKey.trim()),
  )
  const isProcessing = inFlightCount > 0

  useEffect(() => {
    localStorage.setItem("transcriber_script_mode", scriptMode)
  }, [scriptMode])

  // Persist transcript turns to localStorage so recordings stay on screen even after refresh
  useEffect(() => {
    try {
      const compactTurns = turns.slice(-80).map((t, idx, arr) => {
        // Keep playable WAV audio on the most recent 12 turns so localStorage stays well under 5MB
        if (idx < arr.length - 12 && t.audio_wav_b64) {
          const { audio_wav_b64, ...rest } = t
          return rest
        }
        return t
      })
      localStorage.setItem(
        "transcriber_saved_turns_v1",
        JSON.stringify(compactTurns),
      )
    } catch (e) {
      // If quota exceeded, save without audio blobs
      try {
        const textOnly = turns.slice(-80).map(({ audio_wav_b64, ...rest }) => rest)
        localStorage.setItem(
          "transcriber_saved_turns_v1",
          JSON.stringify(textOnly),
        )
      } catch (err) {
        // Ignore
      }
    }
  }, [turns])

  useEffect(() => {
    fetchSpeakers()
      .then((data) => {
        if (data.speakers) setSpeakers(data.speakers)
      })
      .catch(() => {})

    fetchGeminiKeyStatus()
      .then((data) => {
        if (data.has_key) {
          setHasServerKey(true)
          setMaskedServerKey(data.masked_key || "")
        } else if (
          (config.asrEngine || "gemini-cloud") === "gemini-cloud" &&
          !config.geminiApiKey
        ) {
          setShowKeyEditor(true)
        }
      })
      .catch(() => {})
  }, [])

  // Only scroll when a new finalized turn is added — prevents screen jerks during background polls
  useEffect(() => {
    if (turns.length > prevTurnsLenRef.current) {
      if (feedEndRef.current && viewMode === "feed") {
        feedEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" })
      }
    }
    prevTurnsLenRef.current = turns.length
  }, [turns.length, viewMode])

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

  const handleSaveGeminiKey = async (e) => {
    if (e) e.preventDefault()
    const trimmed = keyInput.trim()
    if (!trimmed) return
    setIsTestingKey(true)
    setKeyFeedback(null)
    try {
      const res = await saveAndTestGeminiKey(
        trimmed,
        config.geminiModel || "gemini-3.5-flash-lite",
      )
      if (res.ok) {
        setHasServerKey(true)
        setMaskedServerKey(
          trimmed.length > 10
            ? `${trimmed.slice(0, 6)}...${trimmed.slice(-4)}`
            : "Active",
        )
        setConfig((prev) => ({
          ...prev,
          geminiApiKey: trimmed,
          asrEngine: "gemini-cloud",
        }))
        setKeyFeedback({
          ok: true,
          msg: "Connected to Gemini Cloud Flash",
        })
        setStatusMessage("Gemini Cloud connected — ready to record")
        setTimeout(() => setShowKeyEditor(false), 600)
      } else {
        setKeyFeedback({
          ok: false,
          msg: res.error || "Invalid API Key",
        })
      }
    } catch (err) {
      setKeyFeedback({ ok: false, msg: err.message })
    } finally {
      setIsTestingKey(false)
    }
  }

  const dispatchAudioChunk = useCallback(
    async (base64Data, previewText = "") => {
      const cleanPreview = (previewText || "").trim()
      if (!base64Data && !cleanPreview) return

      const nowTimestamp = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })

      if (!base64Data) return
      setInFlightCount((c) => c + 1)

      try {
        const res = await transcribeConversationChunk(base64Data, {
          engine: activeEngine,
          whisperModel: config.whisperModel || "small",
          allowedLanguages: selectedLangs,
          speakerThreshold: config.speakerThreshold ?? 0.52,
          endpointUrl: config.endpointUrl,
          modelName: config.modelName,
          scriptMode,
          geminiApiKey: config.geminiApiKey || "",
          geminiModel: config.geminiModel || "gemini-3.5-flash-lite",
        })

        if (res.speakers) {
          setSpeakers(res.speakers)
        }

        if (res.turns && res.turns.length > 0) {
          playBlip("speaker")
          const finalizedTurns = res.turns.map((newTurn) => ({
            ...newTurn,
            timestamp: nowTimestamp,
            procTime: res.processing_time,
            isRefining: false,
          }))

          setTurns((prev) => [...prev, ...finalizedTurns])

          const lastTurn = res.turns[res.turns.length - 1]
          const langList = (lastTurn.languages || [])
            .map((l) => l.name)
            .join(" + ")
          setStatusMessage(
            `${lastTurn.speaker_label} · ${langList} (${res.processing_time}s)`,
          )
        }
      } catch (err) {
        console.error("Transcription error:", err)
        setStatusMessage(`Error: ${err.message}`)
        if (err.message && err.message.toLowerCase().includes("api key")) {
          setShowKeyEditor(true)
        }
      } finally {
        setInFlightCount((c) => Math.max(0, c - 1))
      }
    },
    [config, activeEngine, selectedLangs, scriptMode],
  )

  const getPreviewLang = useCallback(() => {
    if (selectedLangs.length === 1) {
      if (selectedLangs[0] === "te") return "te-IN"
      if (selectedLangs[0] === "hi") return "hi-IN"
    }
    return "en-IN"
  }, [selectedLangs])

  // Browser Web Speech API only understands a single language and outputs native script for te-IN/hi-IN,
  // so only enable it when English-only or single-language Native mode is selected.
  const shouldUseBrowserPreview =
    (selectedLangs.length === 1 && selectedLangs[0] === "en") ||
    (selectedLangs.length === 1 && scriptMode === "native")

  const handleToggleRecord = useCallback(async () => {
    if (activeEngine === "gemini-cloud" && !isGeminiReady) {
      setShowKeyEditor(true)
      setStatusMessage("Add your Gemini API key in the sidebar to begin")
      return
    }

    if (isRecording) {
      const audioData = await stopRecording()
      if (
        audioData &&
        ((audioData.base64Data && audioData.durationSec >= 0.25) ||
          audioData.previewText)
      ) {
        dispatchAudioChunk(audioData.base64Data, audioData.previewText || "")
      } else {
        setStatusMessage("Ready to transcribe")
      }
    } else {
      playBlip("ping")
      setStatusMessage("Listening — streaming phrases live")
      await startRecording({
        continuous: false,
        enableBrowserPreview: shouldUseBrowserPreview,
        previewLang: getPreviewLang(),
        onChunkReady: (chunkObj) => {
          if (chunkObj && (chunkObj.base64Data || chunkObj.previewText)) {
            dispatchAudioChunk(
              chunkObj.base64Data,
              chunkObj.previewText || "",
            )
          }
        },
      })
    }
  }, [
    activeEngine,
    isGeminiReady,
    isRecording,
    startRecording,
    stopRecording,
    dispatchAudioChunk,
    getPreviewLang,
    shouldUseBrowserPreview,
  ])

  const handleToggleContinuous = useCallback(async () => {
    if (activeEngine === "gemini-cloud" && !isGeminiReady) {
      setShowKeyEditor(true)
      setStatusMessage("Add your Gemini API key in the sidebar to begin")
      return
    }

    if (isRecording && isContinuous) {
      const leftover = await stopRecording()
      if (
        leftover &&
        ((leftover.base64Data && leftover.durationSec >= 0.25) ||
          leftover.previewText)
      ) {
        dispatchAudioChunk(leftover.base64Data, leftover.previewText || "")
      }
      setStatusMessage("Auto-Listen paused")
    } else {
      if (isRecording) {
        await stopRecording()
      }
      playBlip("ping")
      setStatusMessage("Auto-Listen active — hands-free streaming")
      await startRecording({
        continuous: true,
        enableBrowserPreview: shouldUseBrowserPreview,
        previewLang: getPreviewLang(),
        onChunkReady: (chunkObj) => {
          if (chunkObj && (chunkObj.base64Data || chunkObj.previewText)) {
            dispatchAudioChunk(
              chunkObj.base64Data,
              chunkObj.previewText || "",
            )
          }
        },
      })
    }
  }, [
    activeEngine,
    isGeminiReady,
    isRecording,
    isContinuous,
    startRecording,
    stopRecording,
    dispatchAudioChunk,
    getPreviewLang,
    shouldUseBrowserPreview,
  ])

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
          setStatusMessage(`Enrolling ${enrollingName.trim()}...`)
          const res = await enrollSpeakerAPI(
            enrollingName.trim(),
            audioData.base64Data,
          )
          if (res.speakers) setSpeakers(res.speakers)
          setStatusMessage(`Enrolled speaker "${enrollingName.trim()}"`)
          setEnrollingName("")
          setShowEnrollInput(false)
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
      setStatusMessage("Speaker voiceprints cleared")
    } catch (err) {
      console.error(err)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ""
    try {
      setStatusMessage(`Processing "${file.name}"...`)
      const decoded = await decodeAudioFileToBase64(file)
      if (decoded && decoded.base64Data) {
        dispatchAudioChunk(decoded.base64Data)
      }
    } catch (err) {
      setStatusMessage(`File error: ${err.message}`)
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
            ? `\n  [Meaning: ${t.english_translation}]`
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
    setStatusMessage("Copied full transcript to clipboard")
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
    if (!turn) return
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause()
      audioPlayerRef.current = null
    }
    if (turn.audio_wav_b64) {
      const player = new Audio(`data:audio/wav;base64,${turn.audio_wav_b64}`)
      audioPlayerRef.current = player
      player.play().catch(() => {})
      return
    }
    if (!turn.text) return
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

  const latestTurn = turns.length > 0 ? turns[turns.length - 1] : null
  const showSubtitleTranslation =
    !livePreviewText &&
    latestTurn &&
    config.showEnglishMeaning !== false &&
    latestTurn.english_translation &&
    latestTurn.english_translation.trim() !== "" &&
    latestTurn.english_translation.trim().toLowerCase() !==
      (latestTurn.text || "").trim().toLowerCase()

  return (
    <div className="pro-studio-layout">
      {/* Left Control Sidebar */}
      <aside className="pro-sidebar">
        <div className="pro-sidebar-brand">
          <div className="pro-logo-mark">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" x2="12" y1="19" y2="22" />
            </svg>
          </div>
          <div>
            <div className="pro-brand-name">Voice Studio</div>
            <div className="pro-brand-sub">Multilingual Diarizer</div>
          </div>
        </div>

        {/* Engine Section */}
        <div className="pro-sidebar-section">
          <div className="pro-section-header">
            <span>Transcription Engine</span>
            <span
              className={`pro-status-dot ${
                activeEngine === "gemini-cloud"
                  ? isGeminiReady
                    ? "online"
                    : "warning"
                  : "online"
              }`}
            />
          </div>
          <div className="pro-segmented">
            <button
              className={`pro-seg-btn ${activeEngine === "gemini-cloud" ? "active" : ""}`}
              onClick={() =>
                setConfig((prev) => ({ ...prev, asrEngine: "gemini-cloud" }))
              }
            >
              Gemini 3.5 Flash
            </button>
            <button
              className={`pro-seg-btn ${activeEngine !== "gemini-cloud" ? "active" : ""}`}
              onClick={() =>
                setConfig((prev) => ({ ...prev, asrEngine: "gemma4-audio" }))
              }
            >
              Local Offline
            </button>
          </div>

          {activeEngine === "gemini-cloud" && (
            <div className="pro-key-box">
              <div className="pro-key-status-row">
                <span className="pro-key-label">
                  {isGeminiReady
                    ? `API Key (${maskedServerKey || "Configured"})`
                    : "API Key Required"}
                </span>
                <button
                  type="button"
                  className="pro-link-btn"
                  onClick={() => setShowKeyEditor((v) => !v)}
                >
                  {showKeyEditor ? "Hide" : isGeminiReady ? "Change" : "Add Key"}
                </button>
              </div>

              {(showKeyEditor || !isGeminiReady) && (
                <form onSubmit={handleSaveGeminiKey} className="pro-key-form">
                  <input
                    type="password"
                    className="pro-input"
                    value={keyInput}
                    onChange={(e) => setKeyInput(e.target.value)}
                    placeholder="Paste AIzaSy... Gemini key"
                  />
                  <div className="pro-key-actions">
                    <a
                      href="https://aistudio.google.com/apikey"
                      target="_blank"
                      rel="noreferrer"
                      className="pro-ext-link"
                    >
                      Get free key ↗
                    </a>
                    <button
                      type="submit"
                      className="pro-btn-primary-sm"
                      disabled={isTestingKey || !keyInput.trim()}
                    >
                      {isTestingKey ? "Verifying..." : "Connect"}
                    </button>
                  </div>
                  {keyFeedback && (
                    <div
                      className={`pro-key-feedback ${
                        keyFeedback.ok ? "ok" : "err"
                      }`}
                    >
                      {keyFeedback.msg}
                    </div>
                  )}
                </form>
              )}
            </div>
          )}
        </div>

        {/* Output Script Section */}
        <div className="pro-sidebar-section">
          <div className="pro-section-header">
            <span>Output Script</span>
          </div>
          <div className="pro-segmented">
            <button
              className={`pro-seg-btn ${scriptMode === "roman" ? "active" : ""}`}
              onClick={() => setScriptMode("roman")}
            >
              English Script (Aa)
            </button>
            <button
              className={`pro-seg-btn ${scriptMode === "native" ? "active" : ""}`}
              onClick={() => setScriptMode("native")}
            >
              Native (ఆ / अ)
            </button>
          </div>
          <div className="pro-section-caption">
            {scriptMode === "roman"
              ? "Writes Telugu, Hindi & English using English alphabet (Tanglish / Hinglish)."
              : "Writes Telugu in తెలుగు, Hindi in हिन्दी, and English in Latin."}
          </div>
        </div>

        {/* Active Languages Section */}
        <div className="pro-sidebar-section">
          <div className="pro-section-header">
            <span>Spoken Languages</span>
          </div>
          <div className="pro-lang-grid">
            <button
              className={`pro-lang-option full ${isTriadActive ? "active" : ""}`}
              onClick={() => setSelectedLangs(["te", "en", "hi"])}
            >
              <span className="pro-lang-dots">
                <i style={{ background: "#10b981" }} />
                <i style={{ background: "#6366f1" }} />
                <i style={{ background: "#f59e0b" }} />
              </span>
              <span>Auto-Switch (Telugu + English + Hindi)</span>
            </button>
            <button
              className={`pro-lang-option ${
                selectedLangs.length === 1 && selectedLangs[0] === "te"
                  ? "active"
                  : ""
              }`}
              onClick={() => setSelectedLangs(["te"])}
            >
              Telugu
            </button>
            <button
              className={`pro-lang-option ${
                selectedLangs.length === 1 && selectedLangs[0] === "en"
                  ? "active"
                  : ""
              }`}
              onClick={() => setSelectedLangs(["en"])}
            >
              English
            </button>
            <button
              className={`pro-lang-option ${
                selectedLangs.length === 1 && selectedLangs[0] === "hi"
                  ? "active"
                  : ""
              }`}
              onClick={() => setSelectedLangs(["hi"])}
            >
              Hindi
            </button>
          </div>
        </div>

        {/* Speakers Roster Section */}
        <div className="pro-sidebar-section pro-speakers-section">
          <div className="pro-section-header">
            <span>Speakers ({speakers.length})</span>
            <div style={{ display: "flex", gap: "8px" }}>
              <button
                type="button"
                className="pro-link-btn"
                onClick={() => setShowEnrollInput((v) => !v)}
              >
                + Enroll
              </button>
              {speakers.length > 0 && (
                <button
                  type="button"
                  className="pro-link-btn muted"
                  onClick={handleResetSpeakers}
                >
                  Reset
                </button>
              )}
            </div>
          </div>

          {showEnrollInput && (
            <div className="pro-enroll-box">
              <input
                type="text"
                className="pro-input"
                placeholder="Speaker name (e.g. Sandeep)"
                value={enrollingName}
                onChange={(e) => setEnrollingName(e.target.value)}
              />
              <button
                type="button"
                className={`pro-btn-primary-sm ${
                  isEnrollingRecord ? "recording" : ""
                }`}
                disabled={!enrollingName.trim()}
                onClick={handleEnrollToggle}
              >
                {isEnrollingRecord ? "Stop & Save Sample" : "Record 3s Voice"}
              </button>
            </div>
          )}

          <div className="pro-speaker-list">
            {speakers.length === 0 ? (
              <div className="pro-empty-speakers">
                Voices are automatically fingerprinted and grouped as you speak.
              </div>
            ) : (
              speakers.map((spk) =>
                editingSpeakerId === spk.id ? (
                  <div key={spk.id} className="pro-speaker-edit-row">
                    <input
                      type="text"
                      className="pro-input"
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleSaveRename(spk.id)
                        if (e.key === "Escape") setEditingSpeakerId(null)
                      }}
                      autoFocus
                    />
                    <button
                      className="pro-btn-primary-sm"
                      onClick={() => handleSaveRename(spk.id)}
                    >
                      Save
                    </button>
                  </div>
                ) : (
                  <div
                    key={spk.id}
                    className="pro-speaker-item"
                    onClick={() => {
                      setEditingSpeakerId(spk.id)
                      setRenameValue(spk.label)
                    }}
                    title="Click to rename speaker"
                  >
                    <div
                      className="pro-speaker-avatar"
                      style={{
                        background: `${spk.color}20`,
                        color: spk.color,
                        borderColor: `${spk.color}55`,
                      }}
                    >
                      {getInitials(spk.label)}
                    </div>
                    <div className="pro-speaker-info">
                      <div className="pro-speaker-name">{spk.label}</div>
                      <div className="pro-speaker-meta">
                        {spk.utterance_count || 1} turns · {spk.total_seconds}s
                      </div>
                    </div>
                    <span className="pro-speaker-rename-hint">Rename</span>
                  </div>
                ),
              )
            )}
          </div>
        </div>

        {onOpenSettings && (
          <div className="pro-sidebar-footer">
            <button className="pro-settings-btn" onClick={onOpenSettings}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1Z" />
              </svg>
              <span>Advanced Settings</span>
            </button>
          </div>
        )}
      </aside>

      {/* Main Content Stage */}
      <div className="pro-main-stage">
        {/* Top Bar */}
        <header className="pro-topbar">
          <div className="pro-topbar-left">
            <div className="pro-view-switcher">
              <button
                className={`pro-view-tab ${viewMode === "feed" ? "active" : ""}`}
                onClick={() => setViewMode("feed")}
              >
                Conversation Feed
              </button>
              <button
                className={`pro-view-tab ${
                  viewMode === "document" ? "active" : ""
                }`}
                onClick={() => setViewMode("document")}
              >
                Document View
              </button>
            </div>
            <span className="pro-session-counter">
              {turns.length} {turns.length === 1 ? "turn" : "turns"}
            </span>
          </div>

          <div className="pro-topbar-actions">
            <button
              className="pro-Button-ghost"
              onClick={handleCopyAll}
              disabled={turns.length === 0}
            >
              {copiedId === "all" ? "Copied" : "Copy All"}
            </button>
            <button
              className="pro-Button-ghost"
              onClick={handleExportTxt}
              disabled={turns.length === 0}
            >
              Export .txt
            </button>
            <button
              className="pro-Button-ghost danger"
              onClick={() => {
                setTurns([])
                try {
                  localStorage.removeItem("transcriber_saved_turns_v1")
                } catch (e) {
                  // Ignore
                }
              }}
              disabled={turns.length === 0}
            >
              Clear
            </button>
          </div>
        </header>

        {/* Workspace Scroll Area */}
        <main className="pro-canvas">
          {micError && (
            <div className="pro-alert-error">
              Microphone Error: {micError}
            </div>
          )}

          {turns.length === 0 && !isProcessing && !isRecording ? (
            <div className="pro-empty-hero">
              <div className="pro-empty-icon">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" x2="12" y1="19" y2="22" />
                </svg>
              </div>
              <h2>Start speaking in Telugu, English, or Hindi</h2>
              <p>
                Switch languages mid-sentence without changing any settings.
                Phrases stream live as you speak, automatically identifying each
                speaker and writing in{" "}
                <strong>
                  {scriptMode === "roman"
                    ? "English Script (Romanized)"
                    : "Native Script"}
                </strong>
                .
              </p>
              <div className="pro-feature-pills">
                <span>Live Phrase Streaming</span>
                <span>3D-Speaker Voiceprints</span>
                <span>Mid-Sentence Code-Switching</span>
              </div>
            </div>
          ) : viewMode === "document" ? (
            <div className="pro-doc-sheet">
              <div className="pro-doc-header">
                <span>Transcript Document</span>
                <button className="pro-btn-primary-sm" onClick={handleCopyAll}>
                  {copiedId === "all" ? "Copied" : "Copy Text"}
                </button>
              </div>
              <div className="pro-doc-content">
                {turns.map((turn, idx) => (
                  <div key={turn.id || idx} className="pro-doc-block">
                    <div className="pro-doc-meta">
                      <span
                        className="pro-doc-speaker"
                        style={{ color: turn.speaker_color || "#818cf8" }}
                      >
                        {turn.speaker_label}
                      </span>
                      <span className="pro-doc-langs">
                        {(turn.languages || []).map((l) => l.name).join(" / ")}
                      </span>
                      <span className="pro-doc-time">{turn.timestamp}</span>
                    </div>
                    <div className="pro-doc-text">{turn.text}</div>
                    {config.showEnglishMeaning !== false &&
                      turn.english_translation &&
                      turn.english_translation.toLowerCase() !==
                        turn.text.toLowerCase() && (
                        <div className="pro-doc-translation">
                          {turn.english_translation}
                        </div>
                      )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="pro-feed-stream">
              {turns.map((turn, idx) => {
                const spkColor = turn.speaker_color || "#6366f1"
                const hasMultiClauses = turn.clauses && turn.clauses.length > 1
                const showMeaning =
                  config.showEnglishMeaning !== false &&
                  turn.english_translation &&
                  turn.english_translation.trim() !== "" &&
                  turn.english_translation.trim().toLowerCase() !==
                    turn.text.trim().toLowerCase()

                return (
                  <article key={turn.id || idx} className="pro-turn-row">
                    <div
                      className="pro-turn-avatar"
                      style={{
                        background: `${spkColor}18`,
                        color: spkColor,
                        borderColor: `${spkColor}45`,
                      }}
                      onClick={() => {
                        setEditingSpeakerId(turn.speaker_id)
                        setRenameValue(turn.speaker_label)
                      }}
                      title="Click to rename speaker in sidebar"
                    >
                      {getInitials(turn.speaker_label)}
                    </div>

                    <div className="pro-turn-body">
                      <div className="pro-turn-meta">
                        <span
                          className="pro-turn-speaker"
                          onClick={() => {
                            setEditingSpeakerId(turn.speaker_id)
                            setRenameValue(turn.speaker_label)
                          }}
                        >
                          {turn.speaker_label}
                        </span>

                        {(turn.languages || []).map((l, i) => {
                          const st = getLangStyle(l.code)
                          return (
                            <span
                              key={l.code + i}
                              className="pro-lang-badge"
                              style={{
                                background: st.bg,
                                color: st.text,
                                borderColor: st.border,
                              }}
                            >
                              {l.name}
                            </span>
                          )
                        })}

                        {turn.isRefining && (
                          <span className="pro-switch-tag">
                            Refining speaker & script...
                          </span>
                        )}

                        {turn.language_switched && (
                          <span className="pro-switch-tag">Code-Switched</span>
                        )}

                        <span className="pro-turn-timestamp">
                          {turn.timestamp}
                        </span>

                        <div className="pro-turn-actions">
                          <button
                            className="pro-icon-btn"
                            onClick={() => handleCopyTurn(turn)}
                            title="Copy utterance"
                          >
                            {copiedId === turn.id ? "Copied" : "Copy"}
                          </button>
                          <button
                            className="pro-icon-btn"
                            onClick={() => handleSpeakTurn(turn)}
                            title={
                              turn.audio_wav_b64
                                ? "Play recorded voice clip"
                                : "Read aloud"
                            }
                          >
                            {turn.audio_wav_b64 ? "▶ Play Recording" : "Listen"}
                          </button>
                        </div>
                      </div>

                      <div className="pro-turn-transcript">
                        {hasMultiClauses ? (
                          turn.clauses.map((c, cIdx) => {
                            const st = getLangStyle(c.lang)
                            return (
                              <span
                                key={cIdx}
                                className="pro-clause-span"
                                style={{ borderBottomColor: st.dot }}
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
                        <div className="pro-turn-translation">
                          <span className="pro-translation-label">
                            English
                          </span>
                          <span>{turn.english_translation}</span>
                        </div>
                      )}
                    </div>
                  </article>
                )
              })}

              <div ref={feedEndRef} />
            </div>
          )}
        </main>

        {/* Pinned Cinema Live Subtitle Stage — stays anchored above dock without shifting layout */}
        {(isRecording || isProcessing || latestTurn) && (
          <section className="pro-subtitle-stage" aria-live="polite">
            <div
              className={`pro-subtitle-box ${
                isRecording ? "is-listening" : ""
              }`}
            >
              <div className="pro-subtitle-meta">
                {isRecording && (
                  <span className="pro-subtitle-live-pill">
                    <span className="pro-live-pulse" />
                    <span>LIVE SUBTITLES</span>
                  </span>
                )}
                {!livePreviewText && latestTurn && (
                  <>
                    <span
                      className="pro-subtitle-speaker"
                      style={{ color: latestTurn.speaker_color || "#818cf8" }}
                    >
                      {latestTurn.speaker_label}
                    </span>
                    {(latestTurn.languages || []).map((l, i) => {
                      const st = getLangStyle(l.code)
                      return (
                        <span
                          key={l.code + i}
                          className="pro-subtitle-lang"
                          style={{
                            background: st.bg,
                            color: st.text,
                            borderColor: st.border,
                          }}
                        >
                          {l.name}
                        </span>
                      )
                    })}
                  </>
                )}
                {isProcessing && (
                  <span className="pro-subtitle-decoding">
                    <span className="pro-spinner" />
                    <span>Decoding speech...</span>
                  </span>
                )}
              </div>

              <div className="pro-subtitle-primary">
                {livePreviewText
                  ? livePreviewText
                  : latestTurn
                    ? latestTurn.text
                    : "Listening to audio — subtitles will appear here as speech is detected..."}
              </div>

              {showSubtitleTranslation && (
                <div className="pro-subtitle-translation">
                  <span className="pro-subtitle-en-tag">EN</span>
                  <span>{latestTurn.english_translation}</span>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Floating Bottom Control Dock */}
        <footer className="pro-dock-wrapper">
          <div className="pro-dock">
            <div className="pro-dock-left">
              <StudioWaveform isRecording={isRecording} analyser={analyser} />
              <span
                className={`pro-dock-status ${isRecording ? "active" : ""}`}
              >
                {statusMessage}
              </span>
            </div>

            <div className="pro-dock-center">
              <button
                className={`pro-record-btn ${
                  isRecording && !isContinuous ? "recording" : ""
                }`}
                onClick={handleToggleRecord}
              >
                {isRecording && !isContinuous ? (
                  <>
                    <span className="pro-stop-icon" />
                    <span>Stop ({formatTimer(recordingSeconds)})</span>
                  </>
                ) : (
                  <>
                    <span className="pro-mic-dot" />
                    <span>Record</span>
                    <kbd className="pro-kbd">Space</kbd>
                  </>
                )}
              </button>

              <button
                className={`pro-secondary-dock-btn ${
                  isContinuous ? "active" : ""
                }`}
                onClick={handleToggleContinuous}
              >
                <span
                  className={`pro-auto-dot ${isContinuous ? "active" : ""}`}
                />
                <span>
                  {isContinuous
                    ? `Auto-Listening (${formatTimer(recordingSeconds)})`
                    : "Auto-Listen"}
                </span>
              </button>
            </div>

            <div className="pro-dock-right">
              <button
                className="pro-upload-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={isRecording}
              >
                Upload Audio
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
