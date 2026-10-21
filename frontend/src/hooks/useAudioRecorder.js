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

import { useState, useRef, useCallback, useEffect } from "react"
import { getMergedSamples, resample, blobToBase64 } from "../utils/audioHelpers"

// Ultra-low-latency mic capture hook supporting:
// 1. Live micro-chunk streaming while speaking (flushes on >=0.42s pause or >=2.6s speech)
// 2. Instant <100ms interim word preview via Web Speech API (while neural diarization + ASR runs)
// 3. Audio file decoding to 16kHz Float32 base64
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [isContinuous, setIsContinuous] = useState(false)
  const [livePreviewText, setLivePreviewText] = useState("")
  const [micError, setMicError] = useState(null)

  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const sourceRef = useRef(null)
  const scriptProcessorRef = useRef(null)
  const streamRef = useRef(null)
  const recordedSamplesRef = useRef([])
  const chunkCallbackRef = useRef(null)
  const recognitionRef = useRef(null)
  const isRecordingRef = useRef(false)

  // Fast VAD state for live micro-chunking
  const vadStateRef = useRef({
    hasSpokenInChunk: false,
    voicedFrames: 0,
    silenceFrames: 0,
    totalFrames: 0,
  })

  const stopBrowserLivePreview = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onend = null
        recognitionRef.current.onresult = null
        recognitionRef.current.onerror = null
        recognitionRef.current.stop()
      } catch (e) {
        // Ignore
      }
      recognitionRef.current = null
    }
    setLivePreviewText("")
  }, [])

  const startBrowserLivePreview = useCallback((langHint = "en-IN") => {
    const SpeechRec =
      window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRec) return

    try {
      const rec = new SpeechRec()
      rec.continuous = true
      rec.interimResults = true
      rec.lang = langHint || "en-IN"

      rec.onresult = (event) => {
        let interim = ""
        let latestFinal = ""
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript
          if (event.results[i].isFinal) {
            latestFinal += transcript + " "
          } else {
            interim += transcript
          }
        }
        const combined = (latestFinal + interim).trim()
        if (combined) {
          setLivePreviewText(combined)
        }
      }

      rec.onerror = () => {
        // Ignore browser speech preview errors silently
      }

      rec.onend = () => {
        if (isRecordingRef.current && recognitionRef.current === rec) {
          try {
            rec.start()
          } catch (e) {
            // Ignore restart race
          }
        }
      }

      rec.start()
      recognitionRef.current = rec
    } catch (e) {
      // Browser preview optional
    }
  }, [])

  const cleanupAudioGraph = useCallback(async () => {
    isRecordingRef.current = false
    stopBrowserLivePreview()

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect()
      scriptProcessorRef.current.onaudioprocess = null
      scriptProcessorRef.current = null
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect()
    }
    if (
      audioContextRef.current &&
      audioContextRef.current.state !== "closed"
    ) {
      try {
        await audioContextRef.current.close()
      } catch (e) {
        // Ignore close error
      }
      audioContextRef.current = null
    }
  }, [stopBrowserLivePreview])

  useEffect(() => {
    return () => {
      cleanupAudioGraph()
    }
  }, [cleanupAudioGraph])

  const encodeSamplesToBase64 = useCallback(async (samplesList, sampleRate) => {
    if (!samplesList || samplesList.length === 0) return null
    const mergedSamples = getMergedSamples(samplesList)
    const targetSampleRate = 16000
    const resampledSamples = resample(
      mergedSamples,
      sampleRate,
      targetSampleRate,
    )
    const rawBlob = new Blob([resampledSamples.buffer], {
      type: "application/octet-stream",
    })
    const base64Data = await blobToBase64(rawBlob)
    return {
      rawBlob,
      base64Data,
      durationSec: resampledSamples.length / targetSampleRate,
    }
  }, [])

  const startRecording = useCallback(
    async (options = {}) => {
      const continuous = Boolean(options.continuous)
      const onChunkReady = options.onChunkReady || null
      chunkCallbackRef.current = onChunkReady

      setMicError(null)
      setLivePreviewText("")
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
        })

        streamRef.current = stream
        isRecordingRef.current = true

        // Start instant <100ms interim word preview in parallel
        startBrowserLivePreview(options.previewLang || "en-IN")

        const AudioContext = window.AudioContext || window.webkitAudioContext
        audioContextRef.current = new AudioContext()
        if (audioContextRef.current.state === "suspended") {
          await audioContextRef.current.resume()
        }
        const actualSampleRate = audioContextRef.current.sampleRate || 48000
        const source = audioContextRef.current.createMediaStreamSource(stream)
        sourceRef.current = source

        analyserRef.current = audioContextRef.current.createAnalyser()
        analyserRef.current.fftSize = 256
        source.connect(analyserRef.current)

        // Use 2048 buffer (~42ms at 48kHz) for snappier VAD cut detection
        const bufferSize = 2048
        const scriptProcessor = audioContextRef.current.createScriptProcessor(
          bufferSize,
          1,
          1,
        )
        scriptProcessorRef.current = scriptProcessor
        recordedSamplesRef.current = []
        vadStateRef.current = {
          hasSpokenInChunk: false,
          voicedFrames: 0,
          silenceFrames: 0,
          totalFrames: 0,
        }

        const callbackDurationSec = bufferSize / actualSampleRate
        // Cut phrase after ~0.42s of pause once speech has started
        const silenceCallbacksForCut = Math.max(
          6,
          Math.round(0.42 / callbackDurationSec),
        )
        // Or auto-stream after ~2.8s of continuous speech so user sees text while still talking!
        const maxCallbacksPerChunk = Math.round(2.8 / callbackDurationSec)
        const preRollFrames = Math.max(4, Math.round(0.3 / callbackDurationSec))

        scriptProcessor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0)
          const copy = new Float32Array(inputData)
          recordedSamplesRef.current.push(copy)

          if (chunkCallbackRef.current) {
            let sumSq = 0
            for (let i = 0; i < copy.length; i++) {
              sumSq += copy[i] * copy[i]
            }
            const rms = Math.sqrt(sumSq / copy.length)
            const st = vadStateRef.current
            st.totalFrames += 1

            if (rms > 0.013) {
              st.voicedFrames += 1
              st.silenceFrames = 0
              if (st.voicedFrames >= 3) {
                st.hasSpokenInChunk = true
              }
            } else {
              st.silenceFrames += 1
            }

            // Trim leading silence so chunks start right before speech onset
            if (
              !st.hasSpokenInChunk &&
              recordedSamplesRef.current.length > preRollFrames
            ) {
              recordedSamplesRef.current.shift()
              st.totalFrames = recordedSamplesRef.current.length
            }

            // Flush immediately on a 0.42s pause, or after 2.8s of speech if at a brief dip
            const pauseReached =
              st.hasSpokenInChunk && st.silenceFrames >= silenceCallbacksForCut
            const maxLenReached =
              st.hasSpokenInChunk &&
              st.totalFrames >= maxCallbacksPerChunk &&
              (rms < 0.025 || st.totalFrames >= Math.round(3.6 / callbackDurationSec))

            if (pauseReached || maxLenReached) {
              // Keep last 2 frames (~85ms) as overlap pre-roll if splitting mid-speech
              const overlap = maxLenReached
                ? recordedSamplesRef.current.slice(-2)
                : []
              const chunkToFlush = recordedSamplesRef.current.slice()
              recordedSamplesRef.current = overlap
              vadStateRef.current = {
                hasSpokenInChunk: false,
                voicedFrames: 0,
                silenceFrames: 0,
                totalFrames: overlap.length,
              }
              encodeSamplesToBase64(chunkToFlush, actualSampleRate)
                .then((res) => {
                  if (
                    res &&
                    res.durationSec >= 0.35 &&
                    chunkCallbackRef.current
                  ) {
                    chunkCallbackRef.current(res)
                  }
                })
                .catch((err) =>
                  console.error("Live chunk encode error:", err),
                )
            }
          }
        }

        source.connect(scriptProcessor)
        scriptProcessor.connect(audioContextRef.current.destination)

        setIsContinuous(continuous)
        setIsRecording(true)
        return true
      } catch (err) {
        console.error("Error accessing microphone:", err)
        const msg =
          err.message ||
          "Microphone access failed (HTTPS required for remote devices)"
        setMicError(msg)
        return false
      }
    },
    [encodeSamplesToBase64, startBrowserLivePreview],
  )

  const stopRecording = useCallback(async () => {
    if (!isRecording) return null

    const hasSpoken = vadStateRef.current.hasSpokenInChunk
    const actualSampleRate = audioContextRef.current?.sampleRate || 16000
    const remainingSamples = recordedSamplesRef.current.slice()
    recordedSamplesRef.current = []

    setIsRecording(false)
    setIsContinuous(false)
    chunkCallbackRef.current = null

    await cleanupAudioGraph()

    if (remainingSamples.length === 0 || !hasSpoken) {
      return null
    }

    try {
      return await encodeSamplesToBase64(remainingSamples, actualSampleRate)
    } catch (err) {
      console.error("Base64 encoding failed:", err)
      return null
    }
  }, [isRecording, cleanupAudioGraph, encodeSamplesToBase64])

  const decodeAudioFileToBase64 = useCallback(async (file) => {
    const arrayBuffer = await file.arrayBuffer()
    const AudioContext = window.AudioContext || window.webkitAudioContext
    const tempCtx = new AudioContext()
    try {
      const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer)
      const channelData = audioBuffer.getChannelData(0)
      const resampled = resample(channelData, audioBuffer.sampleRate, 16000)
      const rawBlob = new Blob([resampled.buffer], {
        type: "application/octet-stream",
      })
      const base64Data = await blobToBase64(rawBlob)
      return {
        rawBlob,
        base64Data,
        durationSec: resampled.length / 16000,
      }
    } finally {
      if (tempCtx.state !== "closed") {
        await tempCtx.close()
      }
    }
  }, [])

  return {
    isRecording,
    isContinuous,
    livePreviewText,
    setLivePreviewText,
    startRecording,
    stopRecording,
    decodeAudioFileToBase64,
    analyser: analyserRef.current,
    micError,
    setMicError,
  }
}
