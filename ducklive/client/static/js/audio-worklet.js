/**
 * DuckLive Audio Worklet Processor
 *
 * Captures audio samples from getUserMedia, buffers them into 20ms chunks,
 * and posts to the main thread for resampling + sending to server.
 */

class DuckLiveAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        // 20ms worth of samples at the current sample rate
        this.samplesNeeded = Math.round(sampleRate * 0.02);
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        // Take first channel (mono)
        const samples = input[0];
        for (let i = 0; i < samples.length; i++) {
            this.buffer.push(samples[i]);
        }

        // When we have enough samples, send a chunk
        while (this.buffer.length >= this.samplesNeeded) {
            const chunk = new Float32Array(this.buffer.splice(0, this.samplesNeeded));
            this.port.postMessage(
                { samples: chunk, sourceSampleRate: sampleRate },
                [chunk.buffer]
            );
        }

        return true;
    }
}

registerProcessor('ducklive-audio-processor', DuckLiveAudioProcessor);
