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

import React from "react"

export default function SettingsOverlay({
  isActive,
  onClose,
  config,
  setConfig,
  onTestConnection,
}) {
  const THEME_COLORS = [
    { name: "ORANGE", value: "#ffa500" },
    { name: "GREEN", value: "#4caf50" },
    { name: "YELLOW", value: "#ffeb3b" },
    { name: "WHITE", value: "#ffffff" },
    { name: "BLUE", value: "#38bdf8" },
    { name: "RED", value: "#ff5252" },
  ]

  const [systemVolume, setSystemVolume] = React.useState(null)

  React.useEffect(() => {
    if (isActive) {
      fetch("/api/volume")
        .then((res) => res.json())
        .then((data) => {
          if (data.volume !== undefined && data.volume !== null) {
            setSystemVolume(data.volume)
          }
        })
        .catch((e) => console.error("Failed to fetch volume", e))
    }
  }, [isActive])

  if (!isActive) return null

  const handleChange = (key, value) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  const handleVolumeChange = async (action) => {
    try {
      const res = await fetch("/api/volume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      })
      const data = await res.json()
      if (data.volume !== undefined && data.volume !== null) {
        setSystemVolume(data.volume)
      }
    } catch (e) {
      console.error("Failed to change volume:", e)
    }
  }

  return (
    <div
      className={`settings-overlay ${isActive ? "active" : ""}`}
      style={{ display: isActive ? "flex" : "none" }}
    >
      <header className="overlay-header">
        <h2>System & AI Engine Settings</h2>
        <button className="overlay-close-btn" onClick={onClose}>
          ✕
        </button>
      </header>
      <div className="overlay-body">
        <div className="form-group">
          <label>Application Mode</label>
          <select
            value={config.appMode || "transcriber"}
            onChange={(e) => handleChange("appMode", e.target.value)}
          >
            <option value="transcriber">
              Multi-Speaker & Code-Switch Transcriber (Telugu / English / Hindi
              + 99 Langs)
            </option>
            <option value="translator">
              Classic 2-Lane Voice Translator Kiosk
            </option>
          </select>
        </div>

        <div className="form-group">
          <label>ASR & Code-Switching Engine</label>
          <select
            value={config.asrEngine || "gemma4-audio"}
            onChange={(e) => handleChange("asrEngine", e.target.value)}
          >
            <option value="gemma4-audio">
              Gemma 4 E2B Multimodal Audio + 3D-Speaker Diarizer (Best for
              mid-sentence TE/HI/EN switching)
            </option>
            <option value="hybrid">
              Hybrid (Faster-Whisper + Gemma 4 Audio Disambiguation +
              3D-Speaker)
            </option>
            <option value="whisper">
              Faster-Whisper 99-Lang + 3D-Speaker Diarizer (Fastest CPU mode)
            </option>
          </select>
        </div>

        {config.asrEngine !== "gemma4-audio" && (
          <div className="form-group">
            <label>Whisper Model Size</label>
            <select
              value={config.whisperModel || "small"}
              onChange={(e) => handleChange("whisperModel", e.target.value)}
            >
              <option value="small">
                small (Recommended balance of speed & Indian script accuracy)
              </option>
              <option value="base">base (Ultra-lightweight)</option>
              <option value="large-v3-turbo">
                large-v3-turbo (Highest accuracy, ~1.6GB download on first use)
              </option>
            </select>
          </div>
        )}

        <div className="form-group">
          <label>
            Speaker Voiceprint Threshold:{" "}
            <span>{config.speakerThreshold ?? 0.56}</span> (Lower = stricter,
            Higher = splits new speakers more easily)
          </label>
          <input
            type="range"
            min="0.40"
            max="0.76"
            step="0.02"
            value={config.speakerThreshold ?? 0.56}
            onChange={(e) =>
              handleChange("speakerThreshold", parseFloat(e.target.value))
            }
          />
        </div>

        <div className="form-row-checkboxes">
          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={config.showEnglishMeaning !== false}
              onChange={(e) =>
                handleChange("showEnglishMeaning", e.target.checked)
              }
            />
            <span className="checkbox-label">
              Show English Meaning Subtitle under Telugu / Hindi / Code-Switched
              Turns
            </span>
          </label>
        </div>

        <div className="form-group" style={{ marginBottom: "6px" }}>
          <label>System Volume</label>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <button
              className="overlay-btn"
              style={{ flex: 1 }}
              onClick={() => handleVolumeChange("down")}
            >
              -
            </button>
            <span
              style={{
                minWidth: "40px",
                textAlign: "center",
                fontWeight: "bold",
              }}
            >
              {systemVolume !== null ? `${systemVolume}%` : "--"}
            </span>
            <button
              className="overlay-btn"
              style={{ flex: 1 }}
              onClick={() => handleVolumeChange("up")}
            >
              +
            </button>
          </div>
        </div>

        <div className="form-group">
          <label>Theme Color</label>
          <div style={{ display: "flex", gap: "10px", marginTop: "5px" }}>
            {THEME_COLORS.map((c) => (
              <button
                key={c.name}
                onClick={() => handleChange("themeColor", c.value)}
                title={c.name}
                style={{
                  width: "30px",
                  height: "30px",
                  borderRadius: "50%",
                  backgroundColor: c.value,
                  border:
                    config.themeColor === c.value
                      ? "2px solid #000"
                      : "2px solid transparent",
                  boxShadow:
                    config.themeColor === c.value ? "0 0 0 2px #fff" : "none",
                  cursor: "pointer",
                  padding: 0,
                }}
              />
            ))}
          </div>
        </div>

        <div className="form-group">
          <label>LiteRT-LM / Local LLM Endpoint</label>
          <div className="input-inline">
            <input
              type="text"
              value={config.endpointUrl}
              onChange={(e) => handleChange("endpointUrl", e.target.value)}
            />
            <button className="overlay-btn btn-sm" onClick={onTestConnection}>
              Test
            </button>
          </div>
        </div>

        <div className="form-group">
          <label>Model Name</label>
          <input
            type="text"
            value={config.modelName}
            onChange={(e) => handleChange("modelName", e.target.value)}
            list="model-suggestions"
          />
          <datalist id="model-suggestions">
            <option value="gemma4-e2b"></option>
            <option value="gemma-4-2b"></option>
          </datalist>
        </div>

        <div className="form-group">
          <label>Visualizer Bars: {config.visualizerBars}</label>
          <input
            type="range"
            min="8"
            max="64"
            step="8"
            value={config.visualizerBars}
            onChange={(e) => handleChange("visualizerBars", e.target.value)}
          />
        </div>
      </div>
    </div>
  )
}
