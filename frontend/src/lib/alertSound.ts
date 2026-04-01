/**
 * P1 alert sound using Web Audio API.
 * Generates a two-tone alarm (high-low-high) — no external audio files needed.
 */

let audioCtx: AudioContext | null = null

function getAudioContext(): AudioContext {
  if (!audioCtx) {
    audioCtx = new AudioContext()
  }
  return audioCtx
}

export function playP1AlertSound() {
  try {
    const ctx = getAudioContext()
    if (ctx.state === "suspended") {
      ctx.resume()
    }

    const now = ctx.currentTime
    const gain = ctx.createGain()
    gain.connect(ctx.destination)
    gain.gain.setValueAtTime(0.3, now)
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.8)

    // Two-tone alarm: 880Hz then 660Hz then 880Hz
    const tones = [
      { freq: 880, start: 0, end: 0.15 },
      { freq: 660, start: 0.18, end: 0.33 },
      { freq: 880, start: 0.36, end: 0.51 },
    ]

    for (const tone of tones) {
      const osc = ctx.createOscillator()
      osc.type = "square"
      osc.frequency.setValueAtTime(tone.freq, now + tone.start)

      const toneGain = ctx.createGain()
      toneGain.gain.setValueAtTime(0, now)
      toneGain.gain.setValueAtTime(0.25, now + tone.start)
      toneGain.gain.setValueAtTime(0.25, now + tone.end - 0.02)
      toneGain.gain.linearRampToValueAtTime(0, now + tone.end)

      osc.connect(toneGain)
      toneGain.connect(gain)
      osc.start(now + tone.start)
      osc.stop(now + tone.end)
    }
  } catch {
    // Audio not available — silently fail
  }
}
