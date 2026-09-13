/**
 * VoiceGuard ML Demo Page Script
 * Dedicated frontend logic for live microphone recording and file uploads.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements - Live Mic
    const recordBtn = document.getElementById('recordBtn');
    const micStatus = document.getElementById('micStatus');
    const recTimer = document.getElementById('recTimer');
    const waveformBars = document.querySelectorAll('.waveform-bar');

    // Elements - File Upload
    const dropzone = document.getElementById('dropzone');
    const audioFileInput = document.getElementById('audioFileInput');
    const selectedFileInfo = document.getElementById('selectedFileInfo');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const analyzeUploadBtn = document.getElementById('analyzeUploadBtn');

    // Elements - Results Panel
    const resultsCard = document.getElementById('resultsCard');
    const resultBanner = document.getElementById('resultBanner');
    const bannerIcon = document.getElementById('bannerIcon');
    const bannerStatus = document.getElementById('bannerStatus');
    const bannerSub = document.getElementById('bannerSub');
    
    const metricClassification = document.getElementById('metricClassification');
    const metricConfidence = document.getElementById('metricConfidence');
    const metricSpoofScore = document.getElementById('metricSpoofScore');
    const metricRisk = document.getElementById('metricRisk');
    
    const sourceBadge = document.getElementById('sourceBadge');
    const transcriptSection = document.getElementById('transcriptSection');
    const transcriptText = document.getElementById('transcriptText');

    // State Variables
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let secondsElapsed = 0;
    let selectedFile = null;

    // ============================================================
    // FLOW 1: LIVE HUMAN VOICE RECORDING
    // ============================================================

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            await startRecording();
        } else {
            await stopRecordingAndAnalyze();
        }
    });

    async function startRecording() {
        try {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.start(100);
            isRecording = true;

            // UI state updates
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '⏹';
            micStatus.textContent = 'Recording in progress... Speak now';
            micStatus.style.color = '#10b981';
            
            waveformBars.forEach(bar => bar.classList.add('active'));

            secondsElapsed = 0;
            recTimer.textContent = '00:00';
            timerInterval = setInterval(() => {
                secondsElapsed++;
                const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
                const secs = String(secondsElapsed % 60).padStart(2, '0');
                recTimer.textContent = `${mins}:${secs}`;

                // Auto stop after 12 seconds
                if (secondsElapsed >= 12) {
                    stopRecordingAndAnalyze();
                }
            }, 1000);

        } catch (err) {
            console.error('Microphone access error:', err);
            alert('Microphone access denied or unavailable: ' + err.message);
        }
    }

    async function stopRecordingAndAnalyze() {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;

        clearInterval(timerInterval);
        recordBtn.classList.remove('recording');
        recordBtn.innerHTML = '🎤';
        micStatus.textContent = 'Processing live recording...';
        micStatus.style.color = '#9ca3af';
        waveformBars.forEach(bar => bar.classList.remove('active'));

        // Wait a small delay for chunks to gather
        setTimeout(async () => {
            const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            try {
                micStatus.textContent = 'Encoding WAV audio...';
                const wavBlob = await convertToWavBlob(rawBlob);
                
                micStatus.textContent = 'Running VoiceGuard ML model...';
                const formData = new FormData();
                formData.append('file', wavBlob, 'live_recording.wav');

                const response = await fetch('/demo/analyze?source=microphone', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || 'Analysis request failed');
                }

                const result = await response.json();
                micStatus.textContent = 'Analysis complete!';
                displayResults(result);

            } catch (err) {
                console.error('Recording analysis error:', err);
                alert('Analysis failed: ' + err.message);
                micStatus.textContent = 'Click to record live voice';
            }
        }, 300);
    }

    // Convert raw browser Blob -> AudioBuffer -> 16kHz 16-bit PCM WAV Blob
    async function convertToWavBlob(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        const targetSampleRate = 16000;
        const offlineCtx = new OfflineAudioContext(1, audioBuffer.duration * targetSampleRate, targetSampleRate);
        const src = offlineCtx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(offlineCtx.destination);
        src.start(0);
        
        const resampled = await offlineCtx.startRendering();
        const pcmData = resampled.getChannelData(0);
        
        const wavBuffer = new ArrayBuffer(44 + pcmData.length * 2);
        const view = new DataView(wavBuffer);

        function writeString(offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + pcmData.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, targetSampleRate, true);
        view.setUint32(28, targetSampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, pcmData.length * 2, true);

        let offset = 44;
        for (let i = 0; i < pcmData.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, pcmData[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        audioCtx.close();
        return new Blob([view], { type: 'audio/wav' });
    }

    // ============================================================
    // FLOW 2: UPLOADED CLONED VOICE FILE
    // ============================================================

    dropzone.addEventListener('click', () => audioFileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        audioFileInput.value = '';
        selectedFileInfo.classList.add('hidden');
        analyzeUploadBtn.disabled = true;
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameDisplay.textContent = file.name + ` (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileInfo.classList.remove('hidden');
        analyzeUploadBtn.disabled = false;
    }

    analyzeUploadBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        try {
            analyzeUploadBtn.disabled = true;
            analyzeUploadBtn.innerHTML = '<div class="spinner"></div> Analyzing Audio...';

            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/demo/analyze?source=upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Upload analysis failed');
            }

            const result = await response.json();
            displayResults(result);

        } catch (err) {
            console.error('Upload error:', err);
            alert('File analysis failed: ' + err.message);
        } finally {
            analyzeUploadBtn.disabled = false;
            analyzeUploadBtn.innerHTML = '⚡ Analyze Voice Authenticity';
        }
    });

    // ============================================================
    // RENDER RESULTS
    // ============================================================

    function displayResults(data) {
        resultsCard.classList.remove('hidden');
        resultsCard.scrollIntoView({ behavior: 'smooth' });

        const isFake = data.is_fake;

        // Result Banner Styling
        if (isFake) {
            resultBanner.className = 'result-banner fake';
            bannerIcon.textContent = '🚨';
            bannerStatus.textContent = 'SYNTHETIC / CLONED VOICE DETECTED';
            bannerSub.textContent = 'High probability of AI voice generation or voice cloning';
        } else {
            resultBanner.className = 'result-banner real';
            bannerIcon.textContent = '🛡️';
            bannerStatus.textContent = 'REAL HUMAN VOICE VERIFIED';
            bannerSub.textContent = 'Voice spectrum matches natural human acoustics';
        }

        // Metrics
        metricClassification.textContent = data.classification;
        metricClassification.style.color = isFake ? '#f43f5e' : '#10b981';

        const confPct = (data.confidence * 100).toFixed(1) + '%';
        metricConfidence.textContent = confPct;

        metricSpoofScore.textContent = data.spoof_score.toFixed(4);

        // Risk Level Badge
        const risk = (data.risk_level || 'low').toLowerCase();
        metricRisk.innerHTML = `<span class="risk-tag ${risk}">${risk.toUpperCase()}</span>`;

        // Source badge
        sourceBadge.textContent = data.source === 'microphone' ? 'Source: Live Microphone' : 'Source: Uploaded File';

        // Transcript
        if (data.transcript && data.transcript.trim().length > 0) {
            transcriptSection.classList.remove('hidden');
            transcriptText.textContent = `"${data.transcript}"`;
        } else {
            transcriptSection.classList.add('hidden');
        }
    }
});
