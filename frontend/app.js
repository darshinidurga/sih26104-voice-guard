document.addEventListener("DOMContentLoaded", () => {

    // ============================================================
    // PROTECTED CALL UI
    // ============================================================

    const startCallBtn = document.getElementById("startCallBtn");
    const stopCallBtn = document.getElementById("stopCallBtn");
    const callStatus = document.getElementById("callStatus");
    const callInfo = document.getElementById("callInfo");
    const micZone = document.getElementById("micZone");

    // ============================================================
    // RESULT UI
    // ============================================================

    const valAuthenticity =
        document.getElementById("valAuthenticity");

    const gaugeAuthenticity =
        document.getElementById("gaugeAuthenticity");

    const valRisk =
        document.getElementById("valRisk");

    const valScamScore =
        document.getElementById("valScamScore");

    const indicatorList =
        document.getElementById("indicatorList");

    const transcriptBox =
        document.getElementById("transcriptBox");

    const challengeModal =
        document.getElementById("challengeModal");

    // ============================================================
    // STATE
    // ============================================================

    let microphoneStream = null;
    let mediaRecorder = null;

    let audioContext = null;
    let analyser = null;
    let animationFrame = null;

    let callStartTime = null;
    let timerInterval = null;

    let analysisTimer = null;
    let isCallActive = false;
    let isAnalyzing = false;

    // Analyze approximately every 3 seconds.
    // This matches the V2-A model's 3-second input.
    const CHUNK_SECONDS = 3;

    // ============================================================
    // START PROTECTED CALL
    // ============================================================

    startCallBtn.addEventListener("click", async () => {

        try {

            // ----------------------------------------------------
            // Request microphone permission
            // ----------------------------------------------------

            microphoneStream =
                await navigator.mediaDevices.getUserMedia({
                    audio: true
                });

            isCallActive = true;

            // ----------------------------------------------------
            // Update UI
            // ----------------------------------------------------

            startCallBtn.style.display = "none";
            stopCallBtn.style.display = "block";

            callStatus.innerHTML = `
                <div class="status-dot active"></div>
                <span>Protected Call Active</span>
            `;

            callInfo.textContent =
                "Microphone: Connected ✓";

            micZone.classList.add("active");

            resetResults();

            transcriptBox.textContent =
                "Listening for protected call audio...";

            // Clear any demo state
            demoBadge.style.display = "none";
            demoError.style.display = "none";
            clearDemoLabel();

            // ----------------------------------------------------
            // Start microphone visualization
            // ----------------------------------------------------

            startMicrophoneMonitor();

            // ----------------------------------------------------
            // Start timer
            // ----------------------------------------------------

            callStartTime = Date.now();

            timerInterval = setInterval(() => {

                if (!isCallActive) return;

                const elapsed =
                    Math.floor(
                        (Date.now() - callStartTime) / 1000
                    );

                const minutes =
                    String(Math.floor(elapsed / 60))
                        .padStart(2, "0");

                const seconds =
                    String(elapsed % 60)
                        .padStart(2, "0");

                callInfo.textContent =
                    `Microphone: Connected ✓ • Call: ${minutes}:${seconds}`;

            }, 1000);

            // ----------------------------------------------------
            // Start real-time audio analysis
            // ----------------------------------------------------

            startRealtimeAnalysis();

        } catch (error) {

            console.error(
                "Microphone error:",
                error
            );

            callStatus.innerHTML = `
                <div class="status-dot"></div>
                <span>Microphone access denied</span>
            `;

            callInfo.textContent =
                "Please allow microphone access in your browser.";

            alert(
                "VoiceGuard needs microphone access to protect the call."
            );
        }
    });

    // ============================================================
    // STOP PROTECTED CALL
    // ============================================================

    stopCallBtn.addEventListener(
        "click",
        stopProtectedCall
    );

    function stopProtectedCall() {

        isCallActive = false;

        // Stop scheduled analysis
        if (analysisTimer) {

            clearTimeout(analysisTimer);

            analysisTimer = null;
        }

        // Stop recorder
        if (mediaRecorder) {

            if (
                mediaRecorder.state !== "inactive"
            ) {
                mediaRecorder.stop();
            }

            mediaRecorder = null;
        }

        // Stop microphone
        if (microphoneStream) {

            microphoneStream
                .getTracks()
                .forEach(track => track.stop());

            microphoneStream = null;
        }

        // Stop audio analyser
        if (audioContext) {

            audioContext.close();

            audioContext = null;
        }

        if (animationFrame) {

            cancelAnimationFrame(animationFrame);

            animationFrame = null;
        }

        // Stop timer
        if (timerInterval) {

            clearInterval(timerInterval);

            timerInterval = null;
        }

        // Reset buttons
        startCallBtn.style.display = "block";
        stopCallBtn.style.display = "none";

        callStatus.innerHTML = `
            <div class="status-dot"></div>
            <span>Ready to protect</span>
        `;

        callInfo.textContent =
            "Microphone: Not connected";

        micZone.classList.remove("active");
        micZone.classList.remove("speaking");

        transcriptBox.textContent =
            "Protected call ended.";

        console.log(
            "VoiceGuard protected call stopped."
        );
    }

    // ============================================================
    // REAL-TIME ANALYSIS
    // ============================================================

    function startRealtimeAnalysis() {

        if (!isCallActive) return;

        analyzeMicrophoneChunk();

    }

    async function analyzeMicrophoneChunk() {

        if (!isCallActive) return;

        if (isAnalyzing) {

            scheduleNextAnalysis();

            return;
        }

        isAnalyzing = true;

        let recorder = null;
        let chunks = [];

        try {

            // ----------------------------------------------------
            // Create MediaRecorder
            // ----------------------------------------------------

            const mimeType =
                getSupportedMimeType();

            if (!mimeType) {

                throw new Error(
                    "Browser does not support a suitable audio recording format."
                );
            }

            recorder =
                new MediaRecorder(
                    microphoneStream,
                    {
                        mimeType: mimeType
                    }
                );

            mediaRecorder = recorder;

            // ----------------------------------------------------
            // Collect audio chunks
            // ----------------------------------------------------

            recorder.ondataavailable = event => {

                if (
                    event.data &&
                    event.data.size > 0
                ) {
                    chunks.push(event.data);
                }
            };

            // ----------------------------------------------------
            // Wait until recording finishes
            // ----------------------------------------------------

            const recordingFinished =
                new Promise((resolve, reject) => {

                    recorder.onstop = resolve;

                    recorder.onerror = event => {
                        reject(event.error);
                    };
                });

            recorder.start();

            console.log(
                "Recording analysis chunk..."
            );

            // Record for approximately 3 seconds
            setTimeout(() => {

                if (
                    recorder.state !== "inactive"
                ) {
                    recorder.stop();
                }

            }, CHUNK_SECONDS * 1000);

            await recordingFinished;

            if (!isCallActive) return;

            // ----------------------------------------------------
            // Build audio Blob
            // ----------------------------------------------------

            const audioBlob =
                new Blob(
                    chunks,
                    { type: mimeType }
                );

            console.log(
                "Audio chunk size:",
                audioBlob.size,
                "bytes"
            );

            // ----------------------------------------------------
            // Send to FastAPI
            // ----------------------------------------------------

            const formData =
                new FormData();

            const extension =
                mimeType.includes("webm")
                    ? "webm"
                    : "ogg";

            formData.append(
                "file",
                audioBlob,
                `voice_chunk.${extension}`
            );

            transcriptBox.textContent =
                "Analyzing voice...";

            const response =
                await fetch(
                    "/analyze",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            if (!response.ok) {

                const errorText =
                    await response.text();

                throw new Error(
                    `Server returned ${response.status}: ${errorText}`
                );
            }

            const result =
                await response.json();

            console.log(
                "VoiceGuard result:",
                result
            );

            // ----------------------------------------------------
            // Update UI
            // ----------------------------------------------------

            updateUI(result);

        } catch (error) {

            console.error(
                "Real-time analysis error:",
                error
            );

            if (isCallActive) {

                transcriptBox.textContent =
                    "Voice analysis temporarily unavailable.";
            }

        } finally {

            isAnalyzing = false;

            if (isCallActive) {

                scheduleNextAnalysis();
            }
        }
    }

    // ============================================================
    // SCHEDULE NEXT ANALYSIS
    // ============================================================

    function scheduleNextAnalysis() {

        if (!isCallActive) return;

        analysisTimer =
            setTimeout(() => {

                analyzeMicrophoneChunk();

            }, 200);
    }

    // ============================================================
    // FIND SUPPORTED RECORDING FORMAT
    // ============================================================

    function getSupportedMimeType() {

        const formats = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/ogg"
        ];

        for (const format of formats) {

            if (
                MediaRecorder.isTypeSupported(format)
            ) {

                console.log(
                    "Using recorder format:",
                    format
                );

                return format;
            }
        }

        return null;
    }

    // ============================================================
    // MICROPHONE VISUALIZATION
    // ============================================================

    function startMicrophoneMonitor() {

        audioContext =
            new (
                window.AudioContext ||
                window.webkitAudioContext
            )();

        const source =
            audioContext.createMediaStreamSource(
                microphoneStream
            );

        analyser =
            audioContext.createAnalyser();

        analyser.fftSize = 2048;

        source.connect(analyser);

        monitorAudio();
    }

    function monitorAudio() {

        if (
            !microphoneStream ||
            !analyser
        ) {
            return;
        }

        const buffer =
            new Uint8Array(
                analyser.frequencyBinCount
            );

        analyser.getByteTimeDomainData(
            buffer
        );

        let sum = 0;

        for (
            let i = 0;
            i < buffer.length;
            i++
        ) {

            const normalized =
                (buffer[i] - 128) / 128;

            sum +=
                normalized * normalized;
        }

        const volume =
            Math.sqrt(
                sum / buffer.length
            );

        if (volume > 0.02) {

            micZone.classList.add(
                "speaking"
            );

        } else {

            micZone.classList.remove(
                "speaking"
            );
        }

        animationFrame =
            requestAnimationFrame(
                monitorAudio
            );
    }

    // ============================================================
    // RESET RESULTS
    // ============================================================

    function resetResults() {

        valAuthenticity.textContent =
            "--";

        valAuthenticity.className =
            "metric-value";

        gaugeAuthenticity.style.width =
            "0%";

        valRisk.textContent =
            "--";

        valRisk.className =
            "metric-value";

        valScamScore.textContent =
            "--";

        indicatorList.innerHTML =
            "";

        transcriptBox.textContent =
            "Listening for protected call audio...";
    }

    // ============================================================
    // UPDATE RESULT UI
    // ============================================================

    function updateUI(data) {

        const spoofScore =
            Number(data.confidence || 0);

        // --------------------------------------------------------
        // Voice authenticity
        // --------------------------------------------------------

        if (data.is_fake) {

            valAuthenticity.textContent =
                `SYNTHETIC VOICE (${(spoofScore * 100).toFixed(1)}%)`;

            valAuthenticity.className =
                "metric-value danger";

            gaugeAuthenticity.style.width =
                `${spoofScore * 100}%`;

            gaugeAuthenticity.style.background =
                "var(--danger)";

        } else {

            const realConfidence =
                (1 - spoofScore);

            valAuthenticity.textContent =
                `REAL VOICE (${(realConfidence * 100).toFixed(1)}%)`;

            valAuthenticity.className =
                "metric-value safe";

            gaugeAuthenticity.style.width =
                `${realConfidence * 100}%`;

            gaugeAuthenticity.style.background =
                "var(--success)";
        }

        // --------------------------------------------------------
        // Risk
        // --------------------------------------------------------

        const risk =
            data.risk_level || "low";

        valRisk.textContent =
            risk.toUpperCase();

        valRisk.className =
            "metric-value";

        if (risk === "high") {

            valRisk.classList.add(
                "danger"
            );

        } else if (risk === "medium") {

            valRisk.classList.add(
                "warning"
            );

        } else {

            valRisk.classList.add(
                "safe"
            );
        }

        // --------------------------------------------------------
        // Scam score
        // --------------------------------------------------------

        const scamScore =
            data.scam_score !== undefined
                ? Number(data.scam_score)
                : 0;

        valScamScore.textContent =
            scamScore.toFixed(2);

        // --------------------------------------------------------
        // Indicators
        // --------------------------------------------------------

        indicatorList.innerHTML = "";

        if (data.urgency_detected) {

            addIndicator(
                "Urgency / Pressure Tactics Detected"
            );
        }

        if (
            data.matched_phrases &&
            data.matched_phrases.length > 0
        ) {

            data.matched_phrases.forEach(
                phrase => {

                    addIndicator(
                        `Flagged Phrase: "${phrase}"`
                    );
                }
            );
        }

        // --------------------------------------------------------
        // Transcript
        // --------------------------------------------------------

        if (data.transcript) {

            transcriptBox.textContent =
                data.transcript;
        } else {

            transcriptBox.textContent =
                "Voice analyzed. No transcript available.";
        }

        // --------------------------------------------------------
        // Verification challenge
        // --------------------------------------------------------

        if (data.trigger_challenge) {

            setTimeout(() => {

                if (isCallActive) {

                    challengeModal.classList.add(
                        "active"
                    );
                }

            }, 500);
        }
    }

    // ============================================================
    // ADD INDICATOR
    // ============================================================

    function addIndicator(text) {

        const li =
            document.createElement("li");

        li.textContent = text;

        indicatorList.appendChild(li);
    }

    // ============================================================
    // CLOSE MODAL
    // ============================================================

    window.closeModal = function () {

        challengeModal.classList.remove(
            "active"
        );
    };

    // ============================================================
    // CONTROLLED DEMO
    // ============================================================

    const demoBadge     = document.getElementById("demoBadge");
    const demoError     = document.getElementById("demoError");
    const demoRealBtn   = document.getElementById("demoRealBtn");
    const demoClonedBtn = document.getElementById("demoClonedBtn");
    const demoScamBtn   = document.getElementById("demoScamBtn");

    // Optional sub-label injected under the authenticity value
    let demoLabel = null;

    function showDemoLabel(text) {
        if (demoLabel && demoLabel.parentNode) {
            demoLabel.parentNode.removeChild(demoLabel);
        }
        demoLabel = document.createElement("span");
        demoLabel.className = "demo-sample-label";
        demoLabel.textContent = text;
        valAuthenticity.insertAdjacentElement("afterend", demoLabel);
    }

    function clearDemoLabel() {
        if (demoLabel && demoLabel.parentNode) {
            demoLabel.parentNode.removeChild(demoLabel);
            demoLabel = null;
        }
    }

    async function runDemo(sample) {
        [demoRealBtn, demoClonedBtn, demoScamBtn].forEach(b => { b.disabled = true; });

        demoBadge.style.display = "none";
        demoError.style.display = "none";
        clearDemoLabel();
        transcriptBox.textContent = "Running controlled demonstration...";
        resetResults();

        try {
            const response = await fetch("/demo/" + sample, { method: "POST" });

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({ detail: response.statusText }));
                demoError.textContent = errBody.detail || response.statusText;
                demoError.style.display = "block";
                transcriptBox.textContent = "Demo sample unavailable. See error above.";
                return;
            }

            const result = await response.json();

            // Reuse the existing updateUI — no code duplication
            updateUI(result);

            // Show the CONTROLLED DEMO badge
            demoBadge.style.display = "block";
            showDemoLabel("Controlled demonstration sample");

        } catch (err) {
            demoError.textContent = "Network error — is the server running?";
            demoError.style.display = "block";
            transcriptBox.textContent = "Demo failed.";
        } finally {
            [demoRealBtn, demoClonedBtn, demoScamBtn].forEach(b => { b.disabled = false; });
        }
    }

    demoRealBtn.addEventListener("click",   () => runDemo("real"));
    demoClonedBtn.addEventListener("click", () => runDemo("cloned"));
    demoScamBtn.addEventListener("click",   () => runDemo("cloned_scam"));

});