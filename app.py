import os
import json
import subprocess
import wave
from vosk import Model, KaldiRecognizer
from pyannote.audio import Pipeline


# HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_PATH = "model"
AUDIO_PATH = "data/call.wav"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Diarization
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization",
#     use_auth_token=True
# )
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")


# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization",
#     
# )


diarization = pipeline(AUDIO_PATH)

segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({"speaker": speaker, "start": float(
        turn.start), "end": float(turn.end)})

# 2) Split audio into segments
seg_files = []
for i, seg in enumerate(segments):
    out = f"{OUT_DIR}/seg_{i}_{seg['speaker']}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seg["start"]), "-to", str(seg["end"]),
        "-i", AUDIO_PATH,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        out
    ]
    subprocess.run(cmd, check=True)
    seg_files.append(out)

# 3) Transcribe segments with Vosk
model = Model(MODEL_PATH)


def transcribe(path):
    wf = wave.open(path, "rb")
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            if r.get("text"):
                parts.append(r["text"])
    final = json.loads(rec.FinalResult())
    if final.get("text"):
        parts.append(final["text"])
    return " ".join(parts)


transcripts = []
for i, seg in enumerate(segments):
    text = transcribe(seg_files[i])
    transcripts.append({
        "speaker": seg["speaker"],
        "start": seg["start"],
        "end": seg["end"],
        "text": text
    })

# 4) Merge final transcript
lines = []
for t in transcripts:
    lines.append(
        f"[{t['start']:.2f} - {t['end']:.2f}] {t['speaker']}: {t['text']}")

final_text = "\n".join(lines)
print(final_text)

with open(f"{OUT_DIR}/transcript.txt", "w") as f:
    f.write(final_text)

print("\nSaved to outputs/transcript.txt")
