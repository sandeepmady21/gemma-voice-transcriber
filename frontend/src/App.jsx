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

import React, { useState, useEffect } from "react"
import TranscriberApp from "./TranscriberApp"
import TranslatorApp from "./TranslatorApp"
import SettingsOverlay from "./components/SettingsOverlay"
import { testConnectionAPI } from "./utils/api"

function App() {
  const [config, setConfig] = useState({
    appMode: localStorage.getItem("appMode") || "transcriber", // "transcriber" | "translator"
    viewportMode: localStorage.getItem("viewportMode") || "desktop", // "desktop" | "kiosk"
    asrEngine: localStorage.getItem("asrEngine") || "gemma4-audio", // "gemma4-audio" | "hybrid" | "whisper"
    whisperModel: localStorage.getItem("whisperModel") || "small",
    speakerThreshold: parseFloat(
      localStorage.getItem("speakerThreshold") || "0.56",
    ),
    showEnglishMeaning:
      localStorage.getItem("showEnglishMeaning") !== "false",
    endpointUrl: "http://localhost:9379/v1",
    modelName: "gemma4-e2b",
    apiKey: "",
    keyboardMode: localStorage.getItem("keyboardMode") || "landscape",
    useProxy: true,
    enableTts: true,
    visualizerBars: 32,
    systemPrompt: "Translator mode",
    themeColor: localStorage.getItem("themeColor") || "#ffa500",
  })

  const [isSettingsOpen, setIsSettingsOpen] = useState(false)

  const testConnection = async () => {
    try {
      await testConnectionAPI(
        config.endpointUrl,
        config.useProxy,
        config.apiKey,
      )
    } catch (err) {
      // Ignored: API test failure
    }
  }

  useEffect(() => {
    testConnection()
  }, [])

  useEffect(() => {
    localStorage.setItem("keyboardMode", config.keyboardMode)
    localStorage.setItem("appMode", config.appMode)
    localStorage.setItem("viewportMode", config.viewportMode)
    localStorage.setItem("asrEngine", config.asrEngine)
    localStorage.setItem("whisperModel", config.whisperModel)
    localStorage.setItem("speakerThreshold", String(config.speakerThreshold))
    localStorage.setItem(
      "showEnglishMeaning",
      String(config.showEnglishMeaning),
    )
  }, [
    config.keyboardMode,
    config.appMode,
    config.viewportMode,
    config.asrEngine,
    config.whisperModel,
    config.speakerThreshold,
    config.showEnglishMeaning,
  ])

  useEffect(() => {
    if (config.themeColor) {
      document.documentElement.style.setProperty(
        "--bg-black",
        config.themeColor,
      )
      localStorage.setItem("themeColor", config.themeColor)
    }
  }, [config.themeColor])

  const isKiosk = config.viewportMode === "kiosk"

  return (
    <div
      className={`app-container ${isKiosk ? "kiosk-mode" : "desktop-mode"}`}
    >
      <div className="config-wrap-right" style={{ opacity: 1 }}>
        <button
          className="config-toggle-btn"
          onClick={() =>
            setConfig((prev) => ({
              ...prev,
              viewportMode: prev.viewportMode === "kiosk" ? "desktop" : "kiosk",
            }))
          }
          title="Toggle Desktop / 480x320 Kiosk Screen"
          style={{ marginRight: "4px" }}
        >
          {isKiosk ? "🖥️ DESKTOP" : "📟 480×320"}
        </button>
        <button
          className="config-toggle-btn"
          onClick={() => setIsSettingsOpen(true)}
          title="Settings"
        >
          ⚙️
        </button>
      </div>

      <div style={{ height: "100%", width: "100%" }}>
        {config.appMode === "translator" ? (
          <TranslatorApp config={config} />
        ) : (
          <TranscriberApp config={config} setConfig={setConfig} />
        )}
      </div>

      <SettingsOverlay
        isActive={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        config={config}
        setConfig={setConfig}
        onTestConnection={testConnection}
      />
    </div>
  )
}

export default App
