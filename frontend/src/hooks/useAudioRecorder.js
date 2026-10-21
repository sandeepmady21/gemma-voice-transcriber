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
  const livePreviewRef = useRef("")
  const consumedResultIdxRef = useRef(0)

  // Fast VAD state for live micro-chunking
  const vadStateRef = useRef({
    hasSpokenInChunk: false,
    voicedFrames: 0,
    silenceFrames: 0,
    totalFrames: 0,
  })

  const takeCurrentPreviewText = useCallback(() => {
    const text = (livePreviewRef.current || "").trim()
    livePreviewRef.current = ""
    setLivePreviewText("")
    return text
  }, [])

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
    livePreviewRef.current = ""
    consumedResultIdxRef.current = 0
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
      consumedResultIdxRef.current = 0

      rec.onresult = (event) => {
        const startIdx = Math.min(
          consumedResultIdxRef.current,
          event.results.length,
        )
        let combined = ""
        for (let i = startIdx; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript
          if (event.results[i].isFinal) {
            combined += transcript + " "
          } else {
            combined += transcript
          }
        }
        const trimmed = combined.trim()
        if (trimmed) {
          livePreviewRef.current = trimmed
          setLivePreviewText(trimmed)
        }
        // Tag the recognition instance with latest result count so chunk flush can advance startIdx
        rec._latestResultLength = event.results.length
        rec._allFinalUpTo =
          event.results.length > 0 &&
          event.results[event.results.length - 1].isFinal
            ? event.results.length
            : Math.max(0, event.results.length - 1)
      }

      rec.onerror = () => {
        // Ignore browser speech preview errors silently
      }

      rec.onend = () => {
        if (isRecordingRef.current && recognitionRef.current === rec) {
          consumedResultIdxRef.current = 0
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
      livePreviewRef.current = ""
      setLivePreviewText("")
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: true,
            channelCount: 1,
          },
        })

        streamRef.current = stream
        isRecordingRef.current = true

        // Start instant <100ms interim word preview in parallel
        if (options.enableBrowserPreview !== false) {
          startBrowserLivePreview(options.previewLang || "en-IN")
        }

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
        // Wait for a natural sentence pause (~0.85s) so words are never cut in half
        const silenceCallbacksForCut = Math.max(
          14,
          Math.round(0.85 / callbackDurationSec),
        )
        // Or stream after ~6.0s of continuous speech at a natural breath dip
        const maxCallbacksPerChunk = Math.round(6.0 / callbackDurationSec)
        const preRollFrames = Math.max(7, Math.round(0.35 / callbackDurationSec))

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

            if (rms > 0.012) {
              st.voicedFrames += 1
              st.silenceFrames = 0
              if (st.voicedFrames >= 4) {
                st.hasSpokenInChunk = true
              }
            } else {
              st.silenceFrames += 1
            }

            // Trim leading silence so chunks start ~0.35s before speech onset
            if (
              !st.hasSpokenInChunk &&
              recordedSamplesRef.current.length > preRollFrames
            ) {
              recordedSamplesRef.current.shift()
              st.totalFrames = recordedSamplesRef.current.length
            }

            const pauseReached =
              st.hasSpokenInChunk &&
              st.voicedFrames >= 6 &&
              st.silenceFrames >= silenceCallbacksForCut
            const maxLenReached =
              st.hasSpokenInChunk &&
              st.totalFrames >= maxCallbacksPerChunk &&
              (rms < 0.016 || st.totalFrames >= Math.round(7.5 / callbackDurationSec))

            if (pauseReached || maxLenReached) {
              const overlap = maxLenReached
                ? recordedSamplesRef.current.slice(-5)
                : recordedSamplesRef.current.slice(-3)
              const chunkToFlush = recordedSamplesRef.current.slice()
              recordedSamplesRef.current = overlap
              vadStateRef.current = {
                hasSpokenInChunk: false,
                voicedFrames: 0,
                silenceFrames: 0,
                totalFrames: overlap.length,
              }

              // Snapshot any live preview words captured for this phrase
              const capturedPreview = takeCurrentPreviewText()
              if (recognitionRef.current && recognitionRef.current._latestResultLength) {
                consumedResultIdxRef.current =
                  recognitionRef.current._latestResultLength
              }

              encodeSamplesToBase64(chunkToFlush, actualSampleRate)
                .then((res) => {
                  if (
                    res &&
                    res.durationSec >= 0.5 &&
                    chunkCallbackRef.current
                  ) {
                    chunkCallbackRef.current({
                      ...res,
                      previewText: capturedPreview,
                    })
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
    [encodeSamplesToBase64, startBrowserLivePreview, takeCurrentPreviewText],
  )

  const stopRecording = useCallback(async () => {
    if (!isRecording) return null

    const capturedPreview = (livePreviewRef.current || "").trim()
    const hasSpoken =
      vadStateRef.current.hasSpokenInChunk || Boolean(capturedPreview)
    const actualSampleRate = audioContextRef.current?.sampleRate || 16000
    const remainingSamples = recordedSamplesRef.current.slice()
    recordedSamplesRef.current = []

    setIsRecording(false)
    setIsContinuous(false)
    chunkCallbackRef.current = null

    await cleanupAudioGraph()

    if (remainingSamples.length === 0 && !capturedPreview) {
      return null
    }
    if (!hasSpoken && remainingSamples.length < Math.round((0.4 * actualSampleRate) / 2048)) {
      return null
    }

    try {
      const encoded = await encodeSamplesToBase64(
        remainingSamples,
        actualSampleRate,
      )
      if (!encoded) {
        return capturedPreview
          ? { base64Data: null, durationSec: 0, previewText: capturedPreview }
          : null
      }
      return {
        ...encoded,
        previewText: capturedPreview,
      }
    } catch (err) {
      console.error("Base64 encoding failed:", err)
      return capturedPreview
        ? { base64Data: null, durationSec: 0, previewText: capturedPreview }
        : null
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
