# Call Transcription using Vosk & Whisper (Offline / Local ASR)

This project performs speech-to-text transcription on call recordings using two ASR engines:

- **Vosk** – fast, lightweight, CPU-friendly (fully offline)  
- **Whisper / WhisperX** – higher accuracy with support for multiple model sizes and optional speaker diarization

---

## Requirements

- Python 3.9+  
- pip  
- Linux (tested on Ubuntu-based systems)  

---

## Setup & Installation

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

pip install vosk pyaudio pydub openai-whisper whisperx ffmpeg-python

sudo apt install ffmpeg

whisperx data/call.wav --diarize --model medium.en --hf_token $HF_TOKEN
