# Wake Word Detector

A completely offline wake word detection and audio recording system that works on both macOS and Raspberry Pi. This project provides two implementations:

1. `wake_word_recorder.py` - Uses Porcupine for wake word detection (requires an API key)
2. `local_wake_word.py` - Uses Vosk for completely offline speech recognition (no API key required)

## Features

- **Offline Wake Word Detection**: Listens for a specified wake word or phrase
- **Automatic Recording**: Starts recording when the wake word is detected
- **Voice Activity Detection**: Automatically stops recording after detecting silence
- **Audio Storage**: Saves recordings as WAV files with timestamps
- **Cross-Platform**: Works on both macOS and Raspberry Pi (and other Linux systems)
- **Customizable**: Configurable wake words, sensitivity, and silence thresholds

## Prerequisites

### For macOS:

- Python 3.6 or higher
- Homebrew (for installing PortAudio)

### For Raspberry Pi:

- Python 3.6 or higher
- Development tools for audio processing

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/wake-word-detector.git
cd wake-word-detector
```

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install system dependencies

#### On macOS:

```bash
brew install portaudio
```

#### On Raspberry Pi:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv libportaudio2 libportaudiocpp0 portaudio19-dev libatlas-base-dev
```

### Step 4: Install Python dependencies

#### For Porcupine implementation (requires API key):

```bash
pip install pvporcupine pyaudio numpy
```

#### For Vosk implementation (completely offline):

```bash
pip install vosk webrtcvad pyaudio numpy setuptools
```

### Step 5: Download Vosk model (for `local_wake_word.py`)

```bash
mkdir -p model
curl -L -o model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip model.zip
mv vosk-model-small-en-us-0.15/* model/
rm -r vosk-model-small-en-us-0.15 model.zip
```

For better accuracy but larger size:

```bash
curl -L -o model.zip https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
```

## Project Structure

After installation, your project directory should look like this:

```
wake-word-detector/
├── README.md
├── local_wake_word.py      # Vosk implementation (no API key)
├── wake_word_recorder.py   # Porcupine implementation (requires API key)
├── model/                  # Vosk model files
│   ├── am/
│   ├── conf/
│   ├── graph/
│   └── ivector/
├── recordings/             # Directory where recordings are saved
│   └── recording_20250422_220735.wav  # Example recording
└── venv/                   # Virtual environment
```

> **Note**: The `recordings` directory will be created automatically when you run either script for the first time.

## Example Commands and Output

### Running the Vosk implementation:

```bash
$ python local_wake_word.py --wake_word "hello computer"
Loading Vosk model from 'model'...
Vosk model loaded successfully
Listening for wake word: 'hello computer'
VAD initialized with mode 3
Audio initialized using device: MacBook Pro Microphone
Listening for wake word. Press Ctrl+C to exit.

# When you say "hello computer":
Wake word detected! Recording started...
# Speak after this...
# When silence is detected:
Recording saved to recordings/recording_20250422_220735.wav
```

### Running the Porcupine implementation:

```bash
$ python wake_word_recorder.py --access_key YOUR_ACCESS_KEY --keyword "hey siri"
Porcupine initialized. Listening for 'hey siri'...
Audio initialized using device: MacBook Pro Microphone
Listening for wake word. Press Ctrl+C to exit.

# When you say "hey siri":
Wake word detected! (hey siri)
Recording started...
# Speak after this...
# When silence is detected:
Recording saved to recordings/recording_20250422_221205.wav
```

## Usage

### Option 1: Using Vosk (Completely Offline)

This implementation doesn't require any API keys and works entirely offline:

```bash
python local_wake_word.py --wake_word "hello computer"
```

#### Example Usage Scenarios

1. **Better Wake Word Detection** (recommended for most users):
```bash
python local_wake_word.py --wake_word "hey computer" --vad_mode 2
```

2. **Debugging Voice Detection**:
```bash
python local_wake_word.py --wake_word "hey computer" --vad_mode 2 --debug
```

3. **Noisy Environment**:
```bash
python local_wake_word.py --wake_word "hey computer" --vad_mode 1
```

4. **Quick Commands** (shorter silence duration):
```bash
python local_wake_word.py --wake_word "hey computer" --silence_duration 1.0
```

5. **Full Dictation** (longer recordings):
```bash
python local_wake_word.py --wake_word "record memo" --silence_duration 3.0
```

### Option 2: Using Porcupine (Requires API Key)

This implementation uses Picovoice Porcupine, which requires a free API key:

1. Get a free Picovoice access key from [Picovoice Console](https://console.picovoice.ai/)
2. Run the script with your access key:

```bash
python wake_word_recorder.py --access_key YOUR_ACCESS_KEY --keyword "hey siri"
```

## Configuration Options

### For local_wake_word.py (Vosk):

```
--model MODEL           Path to Vosk model directory (default: model)
--wake_word WAKE_WORD   Wake word or phrase to listen for (default: "hello computer")
--audio_dir AUDIO_DIR   Directory to save recordings (default: recordings)
--vad_mode {0,1,2,3}    VAD aggressiveness (0=least, 3=most aggressive, default: 3)
--silence_duration SILENCE_DURATION
                        Silence duration to stop recording in seconds (default: 2.0)
--debug                 Enable debug mode with verbose output
```

## Advanced Configuration

### Energy Detection Settings

The local wake word detector uses a combination of WebRTC Voice Activity Detection (VAD) and energy-based detection for improved reliability:

- **Energy Threshold**: The minimum energy level to consider audio as speech (default: 0.001)
- **Recording Energy Threshold**: Higher threshold required to trigger recording (default: 0.002)
- **Energy Normalization**: Automatically adjusts to ambient noise levels

These settings can be adjusted in the code for different environments:

- For noisy environments: Increase thresholds to reduce false positives
- For quiet environments: Decrease thresholds to improve sensitivity
- For hands-free operation: Lower thresholds but increase consecutive detections needed

### Debug Mode Output

When running with `--debug`, you'll see various symbols indicating detection status:

- `.` - Active speech detection (appears periodically during speech)
- `📊 Energy: 0.123456 ✓` - Energy level measurement (✓ = above threshold, ✗ = below threshold)
- `👂 Partial: 'text'` - Partial speech recognition result
- `🔍 Final result: 'text'` - Final recognition result
- `✓ Match: 'wake word'` - Detected a potential wake word match
- `✅ ACCEPTED: Clean match` - Wake word accepted (met confidence threshold)
- `❌ REJECTED: Confidence too low` - Wake word rejected (below confidence threshold)
- `⚡ Potential wake word` - Wake word detected in partial results
- `🎤 WAKE WORD DETECTED` - Wake word confirmed, starting recording
- `📦 Trimming buffer` - Audio buffer management information

### Performance Tuning

Different settings work better for different voices, microphones, and environments:

1. **Wake Word Selection**:
   - Multi-word phrases (e.g. "hey computer") are more reliable than single words
   - Distinctive phrases with hard consonants work better
   - Avoid phrases that sound like common words

2. **VAD Mode Adjustment**:
   - Mode 0-1: Less aggressive, works better in quiet environments
   - Mode 2: Balanced, works well for most users
   - Mode 3: Most aggressive, may miss some speech in noisy environments

3. **Silence Duration**:
   - Shorter (0.5-1.0s): For quick commands
   - Longer (2.0-3.0s): For dictation or longer speech

### For wake_word_recorder.py (Porcupine):

```
--access_key ACCESS_KEY  Porcupine access key (required)
--keyword KEYWORD        Predefined wake word (default: hey siri)
--keyword_path KEYWORD_PATH
                         Path to custom wake word model (.ppn)
--sensitivity SENSITIVITY
                         Detection sensitivity (0-1, default: 0.5)
--audio_dir AUDIO_DIR    Directory to save recordings (default: recordings)
--silence_threshold SILENCE_THRESHOLD
                         Silence threshold (default: 0.03)
--silence_duration SILENCE_DURATION
                         Silence duration to stop recording in seconds (default: 2.0)
```

## How It Works

### Vosk Implementation (`local_wake_word.py`)

1. **Initialization**:
   - Loads a Vosk speech recognition model
   - Sets up WebRTC Voice Activity Detection (VAD)
   - Initializes energy-based voice detection
   - Initializes PyAudio for audio capture

2. **Wake Word Detection**:
   - Continuously captures audio from the microphone
   - Uses dual detection (VAD + energy levels) to identify speech
   - When speech is detected, processes the audio with Vosk
   - Uses multiple strategies to detect wake word:
     - Word sequence matching with confidence scoring
     - Exact phrase detection with word boundaries
     - Context-aware position analysis (start/middle/end of phrase)

3. **Recording Process**:
   - When the wake word is detected, starts recording
   - Maintains a small buffer to capture audio just before the wake word
   - Continues recording while speech is detected
   - Stops recording after detecting silence for the specified duration
   - Saves the recording as a WAV file with a timestamp

4. **Audio Processing Pipeline**:
   - Audio capture → Speech detection → Buffer management
   - Speech recognition → Wake word matching → Confidence scoring
   - Energy verification → Recording → Silence detection → Save

### Porcupine Implementation (`wake_word_recorder.py`)

1. **Initialization**:
   - Sets up Porcupine wake word engine with a keyword
   - Initializes PyAudio for audio capture

2. **Wake Word Detection**:
   - Continuously processes audio frames with Porcupine
   - When the wake word is detected, triggers recording

3. **Recording Process**:
   - Starts a new audio stream for recording
   - Records audio until silence is detected
   - Uses RMS amplitude to determine silence
   - Saves the recording to disk as a WAV file

## Troubleshooting

### PyAudio Installation Issues

If you encounter problems installing PyAudio:

#### On macOS:

```bash
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"
pip install pyaudio
```

#### On Raspberry Pi:

```bash
sudo apt-get install -y portaudio19-dev
pip install pyaudio
```

### Permission Issues

If you encounter permission issues accessing the microphone:

#### On Raspberry Pi:
```bash
sudo usermod -a -G audio $USER
# Log out and log back in for changes to take effect
```

## License

This project is released under the MIT License.

