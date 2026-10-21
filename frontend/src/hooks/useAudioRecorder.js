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

// Mic capture hook supporting:
// 1. Push-to-talk recording (startRecording / stopRecording)
// 2. Continuous hands-free VAD auto-chunking (startContinuousRecording)
// 3. Audio file decoding to 16kHz Float32 base64 (decodeAudioFileToBase64)
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [isContinuous, setIsContinuous] = useState(false)
  const [micError, setMicError] = useState(null)

  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const sourceRef = useRef(null)
  const scriptProcessorRef = useRef(null)
  const streamRef = useRef(null)
  const recordedSamplesRef = useRef([])
  const continuousCallbackRef = useRef(null)

  // VAD state for continuous mode
  const vadStateRef = useRef({
    hasSpokenInChunk: false,
    voicedFrames: 0,
    silenceFrames: 0,
    totalFrames: 0,
  })

  const cleanupAudioGraph = useCallback(async () => {
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
  }, [])

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
      continuousCallbackRef.current = onChunkReady

      setMicError(null)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        })

        streamRef.current = stream

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

        const scriptProcessor = audioContextRef.current.createScriptProcessor(
          4096,
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

        // Each callback is 4096 / actualSampleRate seconds (~85ms at 48kHz)
        const callbackDurationSec = 4096 / actualSampleRate
        const silenceCallbacksForCut = Math.max(
          8,
          Math.round(1.05 / callbackDurationSec),
        ) // ~1.05s pause triggers turn dispatch
        const maxCallbacksPerChunk = Math.round(7.5 / callbackDurationSec) // max ~7.5s before auto-flush

        scriptProcessor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0)
          const copy = new Float32Array(inputData)
          recordedSamplesRef.current.push(copy)

          if (continuous && continuousCallbackRef.current) {
            let sumSq = 0
            for (let i = 0; i < copy.length; i++) {
              sumSq += copy[i] * copy[i]
            }
            const rms = Math.sqrt(sumSq / copy.length)
            const st = vadStateRef.current
            st.totalFrames += 1

            if (rms > 0.015) {
              st.voicedFrames += 1
              st.silenceFrames = 0
              if (st.voicedFrames >= 3) {
                st.hasSpokenInChunk = true
              }
            } else {
              st.silenceFrames += 1
            }

            // Keep only ~0.5s of pre-roll silence if speech hasn't started yet
            if (!st.hasSpokenInChunk && recordedSamplesRef.current.length > 7) {
              recordedSamplesRef.current.shift()
              st.totalFrames = recordedSamplesRef.current.length
            }

            const shouldFlush =
              st.hasSpokenInChunk &&
              (st.silenceFrames >= silenceCallbacksForCut ||
                st.totalFrames >= maxCallbacksPerChunk)

            if (shouldFlush) {
              const chunkToFlush = recordedSamplesRef.current.slice()
              recordedSamplesRef.current = []
              vadStateRef.current = {
                hasSpokenInChunk: false,
                voicedFrames: 0,
                silenceFrames: 0,
                totalFrames: 0,
              }
              encodeSamplesToBase64(chunkToFlush, actualSampleRate)
                .then((res) => {
                  if (res && res.durationSec >= 0.45 && continuousCallbackRef.current) {
                    continuousCallbackRef.current(res)
                  }
                })
                .catch((err) => console.error("Continuous chunk encode error:", err))
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
    [encodeSamplesToBase64],
  )

  const stopRecording = useCallback(async () => {
    if (!isRecording) return null

    const wasContinuous = isContinuous
    const hasSpoken = vadStateRef.current.hasSpokenInChunk
    const actualSampleRate = audioContextRef.current?.sampleRate || 16000
    const remainingSamples = recordedSamplesRef.current.slice()
    recordedSamplesRef.current = []

    setIsRecording(false)
    setIsContinuous(false)
    continuousCallbackRef.current = null

    await cleanupAudioGraph()

    if (remainingSamples.length === 0) {
      return null
    }
    if (wasContinuous && !hasSpoken) {
      return null
    }

    try {
      return await encodeSamplesToBase64(remainingSamples, actualSampleRate)
    } catch (err) {
      console.error("Base64 encoding failed:", err)
      return null
    }
  }, [isRecording, isContinuous, cleanupAudioGraph, encodeSamplesToBase64])

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
    startRecording,
    stopRecording,
    decodeAudioFileToBase64,
    analyser: analyserRef.current,
    micError,
    setMicError,
  }
}
