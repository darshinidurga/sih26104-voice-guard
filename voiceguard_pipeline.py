import os
import sys
import json
import subprocess
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from faster_whisper import WhisperModel

from scam_detector import detect_scam
from risk_engine import calculate_risk


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = os.path.join(
    "ml_rebuild",
    "models",
    "voiceguard_v2a.pth"
)
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"

TARGET_SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_LENGTH = TARGET_SAMPLE_RATE * TARGET_SECONDS

SPOOF_THRESHOLD = 0.90


# ============================================================
# V2-A ATTENTION POOLING
# ============================================================

class AttentionPooling(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        scores = self.attention(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        return torch.sum(
            weights * x,
            dim=1
        )


# ============================================================
# V2-A VOICEGUARD MODEL
# ============================================================

class VoiceGuardV2A(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        self.attention = AttentionPooling(256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        x, _ = self.lstm(x)

        x = self.attention(x)

        return self.classifier(x).squeeze(1)


# ============================================================
# VOICEGUARD PIPELINE
# ============================================================

class VoiceGuardPipeline:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(
            f"Initializing VoiceGuard Pipeline "
            f"on device: {self.device}"
        )

        # ----------------------------------------------------
        # Load V2-A model
        # ----------------------------------------------------

        if not os.path.exists(MODEL_FILE):

            raise FileNotFoundError(
                f"V2 model not found: {MODEL_FILE}"
            )

        print("Loading V2-A voice detector...")

        self.voice_model = VoiceGuardV2A().to(
            self.device
        )

        checkpoint = torch.load(
            MODEL_FILE,
            map_location=self.device,
            weights_only=False
        )

        self.voice_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.voice_model.eval()

        print("V2-A model loaded.")

        # ----------------------------------------------------
        # Wav2Vec2
        # ----------------------------------------------------

        print("Loading Wav2Vec2...")

        self.processor = (
            Wav2Vec2Processor.from_pretrained(
                WAV2VEC_MODEL_NAME
            )
        )

        self.wav2vec = (
            Wav2Vec2Model.from_pretrained(
                WAV2VEC_MODEL_NAME
            ).to(self.device)
        )

        self.wav2vec.eval()

        for parameter in self.wav2vec.parameters():
            parameter.requires_grad = False

        print("Wav2Vec2 loaded.")

        # ----------------------------------------------------
        # Faster Whisper
        # ----------------------------------------------------

        print("Loading Faster-Whisper...")

        compute_type = (
            "float16"
            if torch.cuda.is_available()
            else "int8"
        )

        try:

            self.whisper = WhisperModel(
                "base",
                device=str(self.device),
                compute_type=compute_type
            )

        except Exception as e:

            print(
                f"Whisper CUDA notice: {e}"
            )

            print(
                "Falling back to CPU..."
            )

            self.whisper = WhisperModel(
                "base",
                device="cpu",
                compute_type="float32"
            )

        print("Faster-Whisper loaded.")

        print(
            "\nVoiceGuard Pipeline initialized successfully.\n"
        )


    # ========================================================
    # AUDIO LOADING
    # ========================================================

    def _load_audio(self, audio_path):

        if not os.path.exists(audio_path):

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        try:

            waveform, sr = sf.read(
                audio_path,
                dtype="float32"
            )

            # Stereo → mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Resample if necessary
            if sr != TARGET_SAMPLE_RATE:

                print(
                    f"Resampling audio: "
                    f"{sr} Hz → "
                    f"{TARGET_SAMPLE_RATE} Hz"
                )

                waveform = self._resample(
                    waveform,
                    sr
                )

        except Exception:

            # FFmpeg fallback for MP3 / unsupported formats

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-"
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            waveform = np.frombuffer(
                result.stdout,
                dtype=np.float32
            ).copy()

        waveform = np.asarray(
            waveform,
            dtype=np.float32
        )

        return waveform


    # ========================================================
    # RESAMPLING
    # ========================================================

    def _resample(self, waveform, original_rate):

        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=original_rate,
            target_sr=TARGET_SAMPLE_RATE
        )

        return waveform.astype(
            np.float32
        )


    # ========================================================
    # PREPARE 3-SECOND WINDOW
    # ========================================================

    def _prepare_voice_window(self, waveform):

        # Longer than 3 seconds → crop
        if len(waveform) > TARGET_LENGTH:

            return waveform[
                :TARGET_LENGTH
            ]

        # Shorter than 3 seconds → ZERO PAD
        if len(waveform) < TARGET_LENGTH:

            return np.pad(
                waveform,
                (
                    0,
                    TARGET_LENGTH - len(waveform)
                )
            ).astype(np.float32)

        return waveform


    # ========================================================
    # VOICE SPOOF DETECTION
    # ========================================================

    def _detect_voice(self, waveform):

        waveform_3sec = (
            self._prepare_voice_window(
                waveform
            )
        )

        inputs = self.processor(
            waveform_3sec,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = (
            inputs.input_values.to(
                self.device
            )
        )

        with torch.no_grad():

            # Wav2Vec2 sequence
            features = self.wav2vec(
                input_values=input_values
            ).last_hidden_state

            # V2-A BiLSTM + attention + classifier
            logits = self.voice_model(
                features
            )

            spoof_score = torch.sigmoid(
                logits
            ).item()

        is_fake = (
            spoof_score >= SPOOF_THRESHOLD
        )

        return (
            bool(is_fake),
            float(spoof_score)
        )


    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def process_call(self, audio_path):

        print(
            f"Analyzing: {audio_path}"
        )

        # ----------------------------------------------------
        # Load full audio
        # ----------------------------------------------------

        waveform = self._load_audio(
            audio_path
        )

        duration = (
            len(waveform)
            / TARGET_SAMPLE_RATE
        )

        print(
            f"Audio duration: "
            f"{duration:.2f} seconds"
        )

        # ----------------------------------------------------
        # A. V2 VOICE CLONE DETECTION
        # ----------------------------------------------------

        is_fake, spoof_score = (
            self._detect_voice(
                waveform
            )
        )

        print(
            f"V2 spoof score: "
            f"{spoof_score:.3f}"
        )

        print(
            f"Voice result: "
            f"{'SUSPICIOUS' if is_fake else 'NORMAL'}"
        )

        # ----------------------------------------------------
        # B. SPEECH TRANSCRIPTION
        # ----------------------------------------------------

        print(
            "Running transcription..."
        )

        segments, _ = (
            self.whisper.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        # ----------------------------------------------------
        # C. SCAM DETECTION
        # ----------------------------------------------------

        scam_result = detect_scam(
            transcript
        )

        # ----------------------------------------------------
        # D. RISK FUSION
        # ----------------------------------------------------

        risk_result = calculate_risk(
            voice_is_suspicious=is_fake,
            voice_confidence=spoof_score,
            scam_result=scam_result
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        return {

            "is_fake": bool(is_fake),

            # This is the V2 spoof probability.
            # Frontend should display this as "Spoof Score".
            "confidence": round(
                spoof_score,
                2
            ),

            "transcript": transcript,

            "urgency_detected": bool(
                risk_result[
                    "urgency_detected"
                ]
            ),

            "risk_level": (
                risk_result[
                    "risk_level"
                ]
            ),

            "trigger_challenge": bool(
                risk_result[
                    "trigger_challenge"
                ]
            ),

            "scam_score": round(
                scam_result[
                    "risk_score"
                ],
                2
            ),

            "matched_phrases": (
                scam_result[
                    "matched_phrases"
                ]
            ),

            "voice_threshold": (
                SPOOF_THRESHOLD
            )
        }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    audio_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "test_3sec.wav"
    )

    print("=" * 60)

    print(
        "RUNNING VOICEGUARD V2 UNIFIED PIPELINE"
    )

    print(
        f"Audio: {audio_file}"
    )

    print("=" * 60)

    pipeline = VoiceGuardPipeline()

    result = pipeline.process_call(
        audio_file
    )

    print(
        "\n--- FINAL RESULT ---"
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print("=" * 60)