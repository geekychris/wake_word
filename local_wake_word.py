#!/usr/bin/env python3
# Need to add Levenshtein distance for word similarity checking
"""
Local Wake Word Detector and Audio Recorder

This script uses Vosk for offline speech recognition to detect a wake word,
records audio after detection, and saves it locally. It works completely
offline without requiring any API keys or internet connection.
"""

import argparse
import datetime
import json
import os
import queue
import signal
import sys
import threading
import time
import wave
from threading import Thread, Event

import numpy as np
import pyaudio
import webrtcvad
import Levenshtein
import jellyfish  # For phonetic similarity matching
import logging
from vosk import Model, KaldiRecognizer

class LocalWakeWordRecorder:
    def __init__(self, 
                 model_path,
                 wake_word="hello computer",
                 audio_dir="recordings",
                 sample_rate=16000,
                 vad_mode=3,
                 silence_duration=2.0,
                 debug=False):
        """
        Initialize the wake word detector and recorder.
        
        Args:
            model_path: Path to Vosk model directory
            wake_word: Wake word or phrase to listen for (default: "hello computer")
            audio_dir: Directory to save recordings (default: recordings)
            sample_rate: Audio sample rate (16000 recommended for Vosk)
            vad_mode: WebRTC VAD aggressiveness (0-3)
            silence_duration: Seconds of silence to end recording
            debug: Enable debug output
        """
        self.model_path = model_path
        self.wake_word = wake_word.lower()
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.silence_duration = silence_duration
        self.debug = debug
        # Frame parameters - adjusted for better recognition and responsiveness
        self.frame_duration_ms = 30  # 30ms frames for VAD
        self.recognition_frame_size = int(self.sample_rate * 0.5)  # 500ms chunks for better reliability
        self.vad_frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.buffer_size = self.vad_frame_size * 2  # 16-bit audio
        self.buffer_overlap = int(self.sample_rate * 0.6)  # 600ms overlap between buffers for better detection
        self.max_buffer_size = int(self.sample_rate * 3.5)  # Maximum 3.5 seconds of audio
        # Wake word detection settings - significantly lowered for better responsiveness
        self.confidence_threshold = 0.30  # Minimum confidence for word detection (further lowered for better sensitivity)
        self.phrase_confidence_threshold = 0.25  # Minimum confidence for entire phrase (further lowered for better sensitivity)
        self.partial_confidence_threshold = 0.35  # Higher confidence for partial results (further lowered for better sensitivity)
        self.recognition_timeout = 2.0  # Maximum time to spend on a recognition attempt
        self.cooldown_time = 0.5  # Seconds to wait after recording before detecting again
        self.detection_window_time = 1.0  # Time window for consecutive detections
        self.min_detection_gap = 0.2  # Minimum time between recognition attempts
        self.last_detection_time = 0.0  # Time of the last wake word detection
        self.last_recognition_time = 0.0  # Time of the last recognition attempt
        self.min_wake_word_length = 2  # Minimum number of words in wake phrase to reduce false positives
        self.consecutive_detection_count = 0  # Counter for consecutive detections
        self.last_potential_detection_time = 0.0  # Time of last potential detection
        
        # Energy detection settings - with improved dynamic adjustment
        self.energy_threshold = 0.0004  # Lowered threshold for better sensitivity
        self.recording_energy_threshold = 0.0006  # Lowered threshold for better sensitivity
        self.dynamic_energy_adjustment = True  # Enable dynamic energy threshold adjustment
        self.noise_floor = 0.0003  # Initial noise floor estimate
        self.noise_ceiling = 0.0015  # Upper limit for noise floor estimation
        self.noise_floor_frames = []  # Store silence frames for noise floor estimation
        self.noise_floor_window = 150  # Window size for noise floor estimation
        self.energy_normalization_window = 75  # Window size for energy normalization (increased)
        self.energy_smoothing_window = 15  # Increased window size for better energy smoothing
        self.energy_debug_count = 0  # Counter for energy debug output
        self.current_energy = 0.0  # Current energy level (initialized to avoid reference errors)
        self.smoothed_energy = 0.0  # Smoothed energy level using moving average
        self.energy_adjustment_factor = 1.15  # Adjusted factor to balance sensitivity and reliability
        self.energy_history = []  # Initialize energy history
        self.energy_alpha = 0.75  # Adjusted exponential moving average factor for better responsiveness
        self.recording_multiplier = 1.3  # Multiplier for recording threshold relative to energy threshold
        self.adaptive_energy_adjust = True  # Enable adaptive adjustment
        self.min_energy_threshold = 0.0002  # Minimum energy threshold
        self.energy_adjustment_period = 3.0  # Period between adjustments
        self.last_energy_adjustment_time = time.time()  # Initialize last adjustment time
        wake_words = self.wake_word.split()
        if len(wake_words) < self.min_wake_word_length and len(self.wake_word) < 8:
            print(f"Warning: Short wake word '{self.wake_word}' may result in false positives.")
            print("Consider using a phrase of at least 2-3 words like 'hey computer' for better results.")
        
        # Counters for debug info
        self.vad_positive_count = 0
        self.speech_chunks_processed = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('wake_word_detector.log')
            ]
        )
        self.logger = logging.getLogger('wake_word_detector')
        
        # Create recordings directory if it doesn't exist
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Initialize flags
        self.is_recording = False
        self.shutdown_requested = Event()
        
        # Initialize components
        self._init_vosk()
        self._init_vad()
        self._init_audio()
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _init_vosk(self):
        """Initialize Vosk speech recognition model"""
        try:
            self.logger.info(f"Loading Vosk model from '{self.model_path}'...")
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model directory '{self.model_path}' not found.")
                self.logger.error("Download a model from https://alphacephei.com/vosk/models")
                self.logger.error("and unpack it to the 'model' folder.")
                sys.exit(1)
                
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)  # Enable word timestamps for exact matching
            self.recognizer.SetMaxAlternatives(5)  # Get more recognition alternatives for better matching
            self.logger.info(f"Listening for wake word: '{self.wake_word}'")
            self.logger.info(f"Using a recognition frame size of {self.recognition_frame_size} samples ({self.recognition_frame_size/self.sample_rate:.3f} seconds)")
            if self.debug:
                self.logger.debug("Debug mode enabled - showing all recognized speech")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vosk: {e}", exc_info=True)
            sys.exit(1)
    
    def _init_vad(self):
        """Initialize WebRTC Voice Activity Detector"""
        try:
            self.vad = webrtcvad.Vad(self.vad_mode)
            self.logger.info(f"VAD initialized with mode {self.vad_mode}")
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD: {e}", exc_info=True)
            sys.exit(1)
    
    def _init_audio(self):
        """Initialize PyAudio with retry logic"""
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                self.pa = pyaudio.PyAudio()
                
                # Get available input devices and find a working one
                device_count = self.pa.get_device_count()
                if device_count == 0:
                    raise RuntimeError("No audio devices found")
                
                # Try to get default device info
                try:
                    self.device_info = self.pa.get_default_input_device_info()
                    self.device_index = self.device_info['index']
                except IOError:
                    # Fall back to the first device with input channels if default fails
                    self.logger.warning("Default input device not available, searching for alternatives...")
                    
                    for i in range(device_count):
                        dev_info = self.pa.get_device_info_by_index(i)
                        if dev_info['maxInputChannels'] > 0:
                            self.device_info = dev_info
                            self.device_index = i
                            self.logger.info(f"Using alternative input device: {dev_info['name']}")
                            break
                    else:
                        raise RuntimeError("No working input devices found")
                
                # Open stream for detection
                self.logger.info(f"Attempting to open audio stream (attempt {attempt}/{max_retries})...")
                self.audio_stream = self.pa.open(
                    rate=self.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=self.vad_frame_size,
                    input_device_index=self.device_index
                )
                
                # Test the stream to make sure it's working
                test_data = self.audio_stream.read(self.vad_frame_size, exception_on_overflow=False)
                if len(test_data) < self.vad_frame_size:
                    raise RuntimeError(f"Audio stream test failed - received {len(test_data)} bytes, expected {self.vad_frame_size}")
                
                self.logger.info(f"Audio initialized successfully using device: {self.device_info['name']}")
                
                # Log all available audio devices for troubleshooting
                self.logger.debug("Available audio devices:")
                for i in range(self.pa.get_device_count()):
                    dev_info = self.pa.get_device_info_by_index(i)
                    self.logger.debug(f"  Device {i}: {dev_info['name']}")
                    self.logger.debug(f"    Max Input Channels: {dev_info['maxInputChannels']}")
                    self.logger.debug(f"    Default Sample Rate: {dev_info['defaultSampleRate']}")
                
                # Successfully initialized
                return
                
            except Exception as e:
                self.logger.error(f"Audio initialization attempt {attempt}/{max_retries} failed: {e}", exc_info=True)
                
                # Clean up any partially initialized resources
                if hasattr(self, 'audio_stream') and self.audio_stream:
                    try:
                        self.audio_stream.stop_stream()
                        self.audio_stream.close()
                    except:
                        pass
                    self.audio_stream = None
                
                if hasattr(self, 'pa') and self.pa:
                    try:
                        self.pa.terminate()
                    except:
                        pass
                    self.pa = None
                
                # If we have more attempts, wait before retrying
                if attempt < max_retries:
                    self.logger.info(f"Retrying audio initialization in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Failed to initialize audio after multiple attempts")
                    sys.exit(1)
    def _handle_interrupt(self, sig, frame):
        """Handle keyboard interrupt for graceful shutdown"""
        self.logger.info("\nShutdown requested, cleaning up...")
        self.shutdown_requested.set()
    
    def _cleanup(self):
        """Clean up resources before exit"""
        print("Cleaning up resources...")
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                self.logger.error(f"Error closing audio stream: {e}")
        
        if hasattr(self, 'pa') and self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio: {e}")
    
    def _update_noise_floor(self, energy, is_silence=False):
        """Update noise floor estimate based on silent frames with improved dynamics"""
        if is_silence or energy < self.energy_threshold:
            # Add to noise floor estimation frames
            self.noise_floor_frames.append(energy)
            if len(self.noise_floor_frames) > self.noise_floor_window:
                self.noise_floor_frames = self.noise_floor_frames[-self.noise_floor_window:]
            
            # Update noise floor estimate periodically
            current_time = time.time()
            if current_time - self.last_energy_adjustment_time > self.energy_adjustment_period and len(self.noise_floor_frames) > 20:
                # Calculate new noise floor (median is more robust than mean)
                new_noise_floor = np.median(self.noise_floor_frames)
                # Apply exponential moving average to noise floor updates
                if self.noise_floor > 0:
                    # Use exponential moving average for balanced adaptation
                    # More stable smoothing (0.7 old, 0.3 new)
                    self.noise_floor = 0.7 * self.noise_floor + 0.3 * new_noise_floor
                else:
                    # First estimate
                    self.noise_floor = new_noise_floor
                
                # Cap the noise floor to prevent it from getting too high
                self.noise_floor = min(self.noise_floor, self.noise_ceiling)
                
                # Calculate noise variance and mean for adaptive threshold
                noise_variance = np.var(self.noise_floor_frames)
                noise_mean = np.mean(self.noise_floor_frames)
                
                # Calculate coefficient of variation (normalized variance)
                # Higher variation = more conservative (higher) threshold
                # Lower variation = more aggressive (lower) threshold
                multiplier = 1.5  # Default multiplier
                if noise_mean > 0:
                    variation_coef = noise_variance / noise_mean
                    # Scales between 1.3 (stable) and 2.3 (variable)
                    multiplier = 1.3 + min(1.0, variation_coef * 10)
                
                # Calculate new threshold based on noise floor and multiplier
                new_threshold = max(self.min_energy_threshold, self.noise_floor * multiplier)
                
                # Apply exponential smoothing to threshold updates
                if self.adaptive_energy_adjust and hasattr(self, 'energy_threshold') and self.energy_threshold > 0:
                    # Use faster adaptation for more dynamic response (0.6 old, 0.4 new)
                    self.energy_threshold = 0.6 * self.energy_threshold + 0.4 * new_threshold
                else:
                    self.energy_threshold = new_threshold
                
                # Update recording threshold based on energy threshold
                self.recording_energy_threshold = self.energy_threshold * self.recording_multiplier
                
                if self.debug:
                    self.logger.debug(f"Adjusted energy thresholds - noise floor: {self.noise_floor:.6f}, " +
                                     f"multiplier: {multiplier:.2f}, " +
                                     f"speech threshold: {self.energy_threshold:.6f}, " +
                                     f"recording threshold: {self.recording_energy_threshold:.6f}")
                
                self.last_energy_adjustment_time = current_time
    
    def _calculate_energy(self, audio_data):
        """Calculate the energy level of an audio chunk"""
        try:
            # Convert bytes to numpy array of int16
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to -1.0 to 1.0
            audio_np = audio_np.astype(np.float32) / 32768.0
            
            # Calculate RMS energy
            # Calculate RMS energy
            energy = np.sqrt(np.mean(np.square(audio_np)))
            
            # Track energy history
            if not hasattr(self, 'energy_history') or self.energy_history is None:
                self.energy_history = []
            self.energy_history.append(energy)
            if len(self.energy_history) > self.energy_normalization_window:
                self.energy_history = self.energy_history[-self.energy_normalization_window:]
            # Calculate smoothed energy using a moving average
            if len(self.energy_history) >= self.energy_smoothing_window:
                # Use exponential weighted average for smoother transition
                weights = np.linspace(0.5, 1.0, min(len(self.energy_history), self.energy_smoothing_window))
                weights = weights / np.sum(weights)  # Normalize weights to sum to 1
                recent_frames = self.energy_history[-min(len(self.energy_history), self.energy_smoothing_window):]
                recent_energy = np.sum(np.array(recent_frames) * weights)
                
                if self.smoothed_energy > 0:
                    # Use energy_alpha for consistent exponential smoothing
                    self.smoothed_energy = self.energy_alpha * self.smoothed_energy + (1 - self.energy_alpha) * recent_energy
                else:
                    # Initialize with current average if no previous value
                    self.smoothed_energy = recent_energy
                
                # Update noise floor estimate
                self._update_noise_floor(energy, is_silence=(energy < self.energy_threshold))
            
            return energy
        except Exception as e:
            self.logger.error(f"Error calculating energy: {e}", exc_info=True)
            return 0.0
    
    def _is_speech(self, audio_data):
        """Use VAD and energy detection to determine if frame contains speech"""
        try:
            # First check: WebRTC VAD
            vad_result = self.vad.is_speech(audio_data, self.sample_rate)
            
            # Second check: Energy-based detection
            energy = self._calculate_energy(audio_data)
            # Use a more forgiving energy threshold for detection
            energy_result = energy > (self.energy_threshold * 0.9)
            
            # Combined result - either method can trigger detection
            result = vad_result or energy_result
            # Store current energy level and track debug count
            self.current_energy = energy
            self.energy_debug_count += 1
            
            # More frequent energy level logging in debug mode
            if self.debug and self.energy_debug_count % 10 == 0:
                ratio = energy/self.energy_threshold if self.energy_threshold > 0 else 0
                rec_ratio = energy/self.recording_energy_threshold if self.recording_energy_threshold > 0 else 0
                threshold_status = "✓" if energy > self.energy_threshold else "✗"
                rec_status = "✓" if energy > self.recording_energy_threshold else "✗"
                self.logger.debug(f"Energy: {energy:.6f} {threshold_status} (smoothed: {self.smoothed_energy:.6f}), " +
                                 f"speech threshold: {self.energy_threshold:.6f} ({ratio:.2f}x), " +
                                 f"recording threshold: {self.recording_energy_threshold:.6f} {rec_status} ({rec_ratio:.2f}x)")
            if result:
                self.vad_positive_count += 1
                
                if self.debug and self.energy_debug_count % 20 == 0:
                    energy_str = f"{energy:.6f}"
                    smoothed_str = f"{self.smoothed_energy:.6f}" if hasattr(self, 'smoothed_energy') else "N/A"
                    threshold_str = f"{self.recording_energy_threshold:.6f}"
                    ratio_str = f"{energy/self.recording_energy_threshold:.2f}x" if self.recording_energy_threshold > 0 else "N/A"
                    if energy > self.recording_energy_threshold:
                        print(f"\n📊 Energy: {energy_str} ✓ (Smoothed: {smoothed_str}, Threshold: {threshold_str}, Ratio: {ratio_str}, VAD: {'✓' if vad_result else '✗'})", end="", flush=True)
                    else:
                        print(f"\n📊 Energy: {energy_str} ✗ (Smoothed: {smoothed_str}, Threshold: {threshold_str}, Ratio: {ratio_str}, VAD: {'✓' if vad_result else '✗'})", end="", flush=True)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in speech detection: {e}", exc_info=True)
            return False
    
    def _record_audio(self):
        """Record audio until silence is detected"""
        print("Wake word detected! Recording started...")
        self.is_recording = True
        
        # Open a new recording stream
        recording_stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.vad_frame_size
        )
        
        frames = []
        silence_frames = 0
        silence_frames_threshold = int(self.silence_duration * 1000 / self.frame_duration_ms)
        try:
            # Capture a small buffer of audio before the wake word
            # (we already have this from the detection phase)
            if hasattr(self, 'prebuffer') and self.prebuffer:
                frames.extend(self.prebuffer)
                self.prebuffer = []
            
            while self.is_recording and not self.shutdown_requested.is_set():
                # Read audio chunk
                audio_chunk = recording_stream.read(self.vad_frame_size, exception_on_overflow=False)
                frames.append(audio_chunk)
                
                # Check for voice activity
                if self._is_speech(audio_chunk):
                    silence_frames = 0
                else:
                    silence_frames += 1
                    # Stop recording after silence_duration of silence
                    if silence_frames >= silence_frames_threshold:
                        break
                
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            recording_stream.stop_stream()
            recording_stream.close()
            
            # Save recording if we have data
            if frames:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.audio_dir, f"recording_{timestamp}.wav")
                self._save_audio(frames, filename)
            
            self.is_recording = False
    
    def _save_audio(self, frames, filename):
        """Save recorded audio frames to a WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            print(f"Recording saved to {filename}")
        except Exception as e:
            print(f"Error saving recording: {e}")
    
    def _find_wake_word_match(self, text, result):
        """
        Check if wake word is in the text and validate with word timestamps.
        Returns True if wake word is properly detected with high confidence.
        Includes Levenshtein distance and phonetic similarity for more forgiving word matching.
        """
        # Basic initial check
        if not text:
            return False
            
        wake_words = self.wake_word.split()
        text_words = text.split()
        
        # Quick check for simple matches
        if self.wake_word in text.lower():
            if self.debug:
                print(f"\n📌 Found exact wake word in text: '{text}'")
            return True
        
        # Additional check for specific known wake words like "hello computer"
        if self.wake_word == "hello computer":
            # Check for common variations
            variations = [
                "hello computer", "hello computers", "hello computing",
                "hello compter", "hello computed", "hello compute",
                "hello commuter", "hello compete", 
                "hello complete", "hello completely"
            ]
            
            for variation in variations:
                if variation in text.lower():
                    if self.debug:
                        print(f"\n📌 Found wake word variation '{variation}' in: '{text}'")
                    return True
            
        # Check for word-by-word matches anywhere in the text
        if len(wake_words) == 2 and all(w.lower() in text_words for w in wake_words):
            consecutive_words = False
            for i in range(len(text_words) - 1):
                if text_words[i].lower() == wake_words[0].lower() and text_words[i+1].lower() == wake_words[1].lower():
                    consecutive_words = True
                    break
            
            if consecutive_words:
                if self.debug:
                    print(f"\n📌 Found consecutive wake words in text: '{text}'")
                return True
        
        # Ensure minimal phrase length
        if len(wake_words) < self.min_wake_word_length and len(self.wake_word) < 8:
            if self.debug:
                print(f"\nNote: Short wake phrase '{self.wake_word}' may be less reliable")
        
        # Method 1: Precise word sequence matching with timestamps
        if "words" in result:
            words = result["words"]
            
            if self.debug:
                print(f"\nDetected {len(words)} words with timestamps")
            
            # Scan the entire sequence for exact wake word matches
            sequence_matches = []
            
            for i in range(len(words) - len(wake_words) + 1):
                # Check for potential sequence match starting at position i
                match_data = self._check_sequence_match(words, i, wake_words)
                if match_data["is_match"]:
                    sequence_matches.append(match_data)
            
            # Sort matches by confidence (highest first)
            if sequence_matches:
                sequence_matches.sort(key=lambda x: x["confidence"], reverse=True)
                best_match = sequence_matches[0]
                
                # Add enhanced debugging for confidence scores
                if self.debug:
                    print(f"\n🔍 Found {len(sequence_matches)} potential wake word matches")
                    for i, match in enumerate(sequence_matches[:2]):  # Show top 2 matches
                        print(f"  Match {i+1}: '{match['text']}' (conf: {match['confidence']:.2f})")
                
                    print(f"\n✓ Best match: '{best_match['text']}' (confidence: {best_match['confidence']:.2f})")
                    
                    # Show detailed position information
                    if best_match['is_isolated']:
                        print(f"  Position: {best_match['start_idx']}-{best_match['end_idx']} of {len(words)} words (isolated)")
                    else:
                        print(f"  Position: {best_match['start_idx']}-{best_match['end_idx']} of {len(words)} words")
                        
                    # Show threshold comparison
                    print(f"  Required confidence: {self.phrase_confidence_threshold:.2f} (general) / {self.confidence_threshold:.2f} (embedded)")
                    
                    # Debug word confidences if available
                    if 'word_confidences' in best_match:
                        conf_str = ' '.join([f"{c:.2f}" for c in best_match['word_confidences']])
                        print(f"  Word confidences: {conf_str}")
                    
                if best_match["confidence"] >= self.phrase_confidence_threshold:
                    # Check if this is a standalone phrase or part of a short utterance
                    if len(words) <= len(wake_words) + 2 or best_match["is_isolated"]:
                        if self.debug:
                            print(f"  ✅ ACCEPTED: Clean match")
                        return True
                    else:
                        # If embedded in a longer utterance, require higher confidence
                        if best_match["confidence"] >= self.confidence_threshold:
                            if self.debug:
                                print(f"  ✅ ACCEPTED: High confidence in longer utterance")
                            return True
                        elif self.debug:
                            print(f"  ❌ REJECTED: Confidence too low in longer utterance")
                elif self.debug:
                    print(f"  ❌ REJECTED: Confidence below threshold")
                    
                # Special case: if the text is exactly the wake word, accept it with higher probability
                if text.strip() == self.wake_word:
                    if self.debug:
                        print(f"  ✅ ACCEPTED: Exact match with wake word")
                    return True
        
        # Method 2: Basic text matching (fallback)
        # Method 2: Basic text matching (fallback)
        if self.wake_word in text.lower():
            words = text.split()
            
            # Check for exact phrase match
            joined_wake_words = " ".join(wake_words)
            
            # For short utterances, accept wake word anywhere
            if len(words) <= 6 and joined_wake_words in text.lower():
                if self.debug:
                    print(f"\n📌 Accepting wake word in short utterance: '{text}'")
                return True
                
            # For longer utterances, prefer wake word at beginning or end
            if len(words) > 6:
                if text.lower().startswith(joined_wake_words) or text.lower().endswith(joined_wake_words):
                    if self.debug:
                        print(f"\n📌 Accepting wake word at start/end of utterance: '{text}'")
                    return True
                
                # Otherwise only accept if it's a very strong match
                wake_phrase_count = text.lower().count(joined_wake_words)
                if wake_phrase_count >= 2:  # Multiple occurrences increase probability
                    if self.debug:
                        print(f"\n📌 Accepting multiple wake word occurrences: '{text}'")
                    return True
        
        # Method 3: Check for individual words with correct order
        words = text.lower().split()
        wake_words_lower = [w.lower() for w in wake_words]
        
        # If we have a two-word wake phrase like "hello computer"
        if len(wake_words_lower) == 2:
            first_word_positions = [i for i, w in enumerate(words) if w == wake_words_lower[0] or w.startswith(wake_words_lower[0])]
            second_word_positions = [i for i, w in enumerate(words) if w == wake_words_lower[1] or w.startswith(wake_words_lower[1])]
            
            # Check if words appear in sequence (not necessarily consecutive)
            for pos1 in first_word_positions:
                for pos2 in second_word_positions:
                    if pos1 < pos2 and pos2 - pos1 <= 3:  # Allow up to 2 words between
                        if self.debug:
                            print(f"\n📌 Found wake words in sequence: '{words[pos1]}...{words[pos2]}' in '{text}'")
                        return True
        return False
    
    def _check_sequence_match(self, words, start_idx, wake_words):
        """Check if a sequence of words matches the wake word sequence."""
        result = {
            "is_match": False,
            "confidence": 0.0,
            "text": "",
            "start_time": 0,
            "end_time": 0,
            "is_isolated": False
        }
        
        # Check if sequence matches
        if start_idx + len(wake_words) > len(words):
            return result
            
        # Check each word in sequence
        matched_words = []
        word_confidences = []
        start_time = 0
        end_time = 0
        is_match = True
        
        for i, target_word in enumerate(wake_words):
            idx = start_idx + i
            word_data = words[idx]
            actual_word = word_data["word"].lower()
            target_word_lower = target_word.lower()
            
            # Check for word match - now with Levenshtein distance for more forgiving matching
            if actual_word != target_word_lower:
                # Calculate Levenshtein distance for fuzzy matching
                distance = Levenshtein.distance(actual_word, target_word_lower)
                
                # Add phonetic similarity matching with Soundex
                phonetic_match = False
                try:
                    # Calculate phonetic similarity using Soundex algorithm
                    actual_soundex = jellyfish.soundex(actual_word)
                    target_soundex = jellyfish.soundex(target_word_lower)
                    
                    # Check for exact Soundex match
                    if actual_soundex == target_soundex:
                        phonetic_match = True
                    
                    # If Soundex didn't match but we have a close Levenshtein match
                    # and the words are long enough, consider it a match
                    if not phonetic_match and len(target_word_lower) > 3 and distance <= 1:
                        # Secondary phonetic check with metaphone
                        actual_metaphone = jellyfish.metaphone(actual_word)
                        target_metaphone = jellyfish.metaphone(target_word_lower)
                        
                        # If metaphones are similar, consider it a match
                        if actual_metaphone == target_metaphone or Levenshtein.distance(actual_metaphone, target_metaphone) <= 1:
                            phonetic_match = True
                except Exception as e:
                    # If phonetic matching fails, fall back to just Levenshtein
                    self.logger.warning(f"Phonetic matching error: {e}")
                
                # Allow for slight variations in word detection with enhanced rules
                # For short words (1-3 chars), allow 0 changes
                # For medium words (4-6 chars), allow 1 change
                # For longer words, allow up to 2 changes
                max_allowed_distance = 0
                if len(target_word_lower) >= 4 and len(target_word_lower) <= 6:
                    max_allowed_distance = 1
                elif len(target_word_lower) > 6:
                    max_allowed_distance = 2
                
                # If we have a phonetic match, be more lenient with the Levenshtein distance
                if phonetic_match:
                    max_allowed_distance += 1
                    
                # If the distance is too great and no phonetic match, this isn't a match
                if distance > max_allowed_distance and not phonetic_match:
                    is_match = False
                    break
                
            # Track confidence values
            if "conf" in word_data:
                word_confidences.append(word_data["conf"])
                
            # Add to matched words list
            matched_words.append(actual_word)
        
        if not is_match or not word_confidences:
            return result
            
        # Store word confidences for debugging
        result['word_confidences'] = word_confidences
        # Calculate confidence as weighted average (more weight to first and last words)
        if len(word_confidences) > 1:
            # Give 50% weight to first and last word combined
            confidence = 0.25 * word_confidences[0]  # 25% first word
            confidence += 0.25 * word_confidences[-1]  # 25% last word
            
            # Divide remaining 50% among middle words
            if len(word_confidences) > 2:
                middle_avg = sum(word_confidences[1:-1]) / (len(word_confidences) - 2)
                confidence += 0.5 * middle_avg
            else:
                confidence += 0.5 * (word_confidences[0] + word_confidences[-1]) / 2
        else:
            confidence = word_confidences[0]
        
        # Check if the wake word is isolated within the utterance
        # A wake word is considered isolated if:
        # 1. It's at the beginning or end of the utterance
        # 2. There's sufficient pause before/after it
        is_isolated = False
        
        # Check if it's at the beginning or end of utterance
        if start_idx == 0 or start_idx + len(wake_words) == len(words):
            is_isolated = True
        
        # Check for pauses before/after (if time information is available)
        if not is_isolated and "start" in words[0] and "end" in words[-1]:
            total_utterance_time = words[-1]["end"] - words[0]["start"]
            
            # Calculate pause before wake word (if not at beginning)
            if start_idx > 0 and "end" in words[start_idx-1] and "start" in words[start_idx]:
                pause_before = words[start_idx]["start"] - words[start_idx-1]["end"]
                # If pause is significant (> 15% of utterance), consider it isolated
                if pause_before > 0.15 * total_utterance_time:
                    is_isolated = True
            
            # Calculate pause after wake word (if not at end)
            if start_idx + len(wake_words) < len(words) and "end" in words[start_idx + len(wake_words) - 1] and "start" in words[start_idx + len(wake_words)]:
                pause_after = words[start_idx + len(wake_words)]["start"] - words[start_idx + len(wake_words) - 1]["end"]
                # If pause is significant, consider it isolated
                if pause_after > 0.15 * total_utterance_time:
                    is_isolated = True
        
        # Build result object
        matched_text = " ".join(matched_words)
        
        result = {
            "is_match": True,
            "confidence": confidence,
            "text": matched_text,
            "start_time": start_time,
            "end_time": end_time,
            "is_isolated": is_isolated,
            "start_idx": start_idx,
            "end_idx": start_idx + len(wake_words) - 1
        }
        
        return result
        
    def run(self):
        """Main processing loop for wake word detection"""
        print(f"Listening for wake word: '{self.wake_word}'")
        print("Press Ctrl+C to exit")
        
        # Initialize variables
        audio_buffer = b''
        speechless_frames = 0
        max_speechless_frames = int(1.0 * 1000 / self.frame_duration_ms)  # 1 second of silence
        self.prebuffer = []
        max_prebuffer_frames = 90  # Keep 2700ms of audio before wake word (increased for better detection)
        
        try:
            while not self.shutdown_requested.is_set():
                # Read audio chunk with error handling
                try:
                    audio_data = self.audio_stream.read(self.vad_frame_size, exception_on_overflow=False)
                except Exception as e:
                    self.logger.error(f"Error reading audio data: {e}")
                    time.sleep(0.1)  # Short delay before retry
                    continue
                
                # Maintain a small buffer of recent audio
                self.prebuffer.append(audio_data)
                if len(self.prebuffer) > max_prebuffer_frames:
                    self.prebuffer.pop(0)
                
                # Only process through Vosk if not already recording
                if not self.is_recording:
                    # Check for cooldown period
                    current_time = time.time()
                    if current_time - self.last_detection_time < self.cooldown_time:
                        continue
                    
                    # Check if speech is present first (optimization)
                    is_speech = self._is_speech(audio_data)
                    if is_speech:
                        # Add to buffer
                        audio_buffer += audio_data
                        speechless_frames = 0
                        
                        # Limit buffer size to prevent memory issues
                        if len(audio_buffer) > self.max_buffer_size:
                            if self.debug:
                                print(f"\n📦 Limiting buffer size from {len(audio_buffer)} to {self.max_buffer_size} bytes")
                            audio_buffer = audio_buffer[-self.max_buffer_size:]
                        # Process with Vosk once we have enough data
                        if len(audio_buffer) >= self.recognition_frame_size:
                            # Check if enough time has passed since last recognition attempt
                            current_time = time.time()
                            time_since_last = current_time - self.last_recognition_time
                            if time_since_last < self.min_detection_gap:
                                if self.debug:
                                    self.logger.debug(f"Skipping recognition, only {time_since_last:.3f}s since last attempt")
                                continue
                            
                            # Show buffer size for debugging
                            if self.debug and len(audio_buffer) > 0:
                                buffer_seconds = len(audio_buffer) / self.sample_rate / 2
                                print(f"\n🎧 Processing {len(audio_buffer)} bytes ({buffer_seconds:.2f}s) of audio")
                                # Print energy level compared to thresholds for diagnosis
                                if hasattr(self, 'current_energy'):
                                    energy_ratio = self.current_energy / self.recording_energy_threshold if self.recording_energy_threshold > 0 else 0
                                    smoothed_ratio = self.smoothed_energy / self.recording_energy_threshold if hasattr(self, 'smoothed_energy') and self.recording_energy_threshold > 0 else 0
                                    print(f"  📊 Current energy: {self.current_energy:.6f} ({energy_ratio:.2f}x threshold)")
                                    if hasattr(self, 'smoothed_energy'):
                                        print(f"  📊 Smoothed energy: {self.smoothed_energy:.6f} ({smoothed_ratio:.2f}x threshold)")
                            # Update recognition time
                            self.last_recognition_time = current_time
                            
                            # Check for recognition timeout
                            if current_time - self.last_recognition_time > self.recognition_timeout:
                                # Reset buffer if we've spent too long on this recognition attempt
                                if self.debug:
                                    print("\n⏱️ Recognition timeout - resetting buffer")
                                # Keep a bit more buffer on timeout for better continuity
                                keep_size = min(len(audio_buffer), self.buffer_overlap * 3)
                                audio_buffer = audio_buffer[-keep_size:] if keep_size > 0 else b''
                            
                            # Process buffer with Vosk
                            self.speech_chunks_processed += 1
                            if self.debug:
                                print(f"\n🎙️ Processing chunk #{self.speech_chunks_processed}")
                            # Process final results - try to get a complete utterance
                            recognition_result = self.recognizer.AcceptWaveform(audio_buffer[:self.recognition_frame_size])
                            
                            if self.debug:
                                print(f"  AcceptWaveform result: {recognition_result}")
                            
                            # Get the result - could be final or partial
                            if recognition_result:
                                # This is a final result
                                result_json = self.recognizer.Result()
                                result_data = json.loads(result_json)
                                
                                if 'text' in result_data:
                                    text = result_data.get('text', '').lower()
                                    if text and self.debug:
                                        print(f"\n🔍 Final result: '{text}'")
                                        
                                        # Print the confidence scores if available
                                        if "conf" in result_data:
                                            print(f"  Overall confidence: {result_data['conf']:.2f}")
                                    
                                    # Check for wake word using high-precision matching
                                    wake_word_match = self._find_wake_word_match(text, result_data)
                                    
                                    if self.debug:
                                        if wake_word_match:
                                            print(f"  ✓ WAKE WORD DETECTED in final result")
                                        else:
                                            print(f"  ✗ No wake word match in: '{text}'")
                                    
                                    if wake_word_match:
                                        # Update consecutive detection count
                                        # Update consecutive detection count
                                        if current_time - self.last_potential_detection_time < self.detection_window_time:
                                            self.consecutive_detection_count += 1
                                        else:
                                            # Reset if too much time has passed
                                            self.consecutive_detection_count = 1
                                        
                                        self.last_potential_detection_time = current_time
                                        
                                        # Check energy level - improved for better sensitivity
                                        energy_sufficient = False
                                        # Use smoothed energy if available, otherwise current energy
                                        if hasattr(self, 'smoothed_energy') and self.smoothed_energy > 0:
                                            # Use smoothed energy with a 20% margin for better sensitivity
                                            energy_sufficient = self.smoothed_energy >= (self.recording_energy_threshold * 0.8)
                                        else:
                                            # Fall back to current energy with 15% margin
                                            energy_sufficient = self.current_energy >= (self.recording_energy_threshold * 0.85)
                                        
                                        # Print debug information about energy levels
                                        if self.debug:
                                            energy_ratio = self.current_energy / self.recording_energy_threshold
                                            smoothed_ratio = self.smoothed_energy / self.recording_energy_threshold if hasattr(self, 'smoothed_energy') else 0
                                            speech_ratio = self.current_energy / self.energy_threshold
                                            threshold_status = "✓" if energy_sufficient else "✗"
                                            
                                            if energy_sufficient:
                                                print(f"  Energy level: {self.current_energy:.6f} ✓ (Rec threshold: {self.recording_energy_threshold:.6f}, Ratio: {energy_ratio:.2f}x)")
                                                print(f"  Speech threshold: {self.energy_threshold:.6f}, Ratio: {speech_ratio:.2f}x")
                                                if hasattr(self, 'smoothed_energy'):
                                                    print(f"  Smoothed energy: {self.smoothed_energy:.6f} ✓ (Ratio: {smoothed_ratio:.2f}x)")
                                            else:
                                                print(f"  Energy level: {self.current_energy:.6f} ✗ (Rec threshold: {self.recording_energy_threshold:.6f}, Ratio: {energy_ratio:.2f}x)")
                                                print(f"  Speech threshold: {self.energy_threshold:.6f}, Ratio: {speech_ratio:.2f}x")
                                                if hasattr(self, 'smoothed_energy'):
                                                    print(f"  Smoothed energy: {self.smoothed_energy:.6f} ✗ (Ratio: {smoothed_ratio:.2f}x)")
                                        
                                        # Print wake word detection notice
                                        print(f"\n🎤 WAKE WORD DETECTED: '{text}'")
                                        
                                        # Only trigger if energy is sufficient
                                        if energy_sufficient:
                                            print(f"   Starting recording...")
                                            # Reset counter
                                            self.consecutive_detection_count = 0
                                            self.last_detection_time = time.time()
                                            try:
                                                record_thread = Thread(target=self._record_audio)
                                                record_thread.start()
                                                record_thread.join()
                                            except Exception as e:
                                                self.logger.error(f"Error in recording thread: {e}", exc_info=True)
                                            finally:
                                                audio_buffer = b''
                                                continue
                                        elif self.debug:
                                            print(f"  ⚠️ Wake word detected but energy too low for recording")
                            else:
                                # Not a final result, get partial result
                                partial_json = self.recognizer.PartialResult()
                                try:
                                    partial_result = json.loads(partial_json)
                                    
                                    if 'partial' in partial_result:
                                        partial_text = partial_result.get('partial', '').lower()
                                        
                                        # Improve partial result handling
                                        if partial_text:
                                            # Clean up partial text - strip common repeated words
                                            words = partial_text.split()
                                            
                                            # Check for repetitive words which can indicate recognition errors
                                            repeated = False
                                            repeated_count = 0
                                            
                                            # Check for word repetition which often indicates poor recognition
                                            if len(words) > 1:
                                                for i in range(1, len(words)):
                                                    if words[i] == words[i-1]:
                                                        repeated_count += 1
                                                        if repeated_count > 3:  # More than 3 repeated words
                                                            repeated = True
                                                            break
                                            if repeated and self.debug:
                                                print(f"\n⚠️ Repetitive partial result detected - filtering")
                                                # Skip this partial result
                                                continue
                                            if self.debug and len(partial_text) < 50:  # Limit debug output
                                                print(f"\n👂 Partial: '{partial_text}'")
                                            
                                            # More sophisticated wake word check - look for exact matches
                                            wake_words = self.wake_word.split()
                                            found_exact = False
                                            
                                            # For multi-word wake phrases, check for exact sequence
                                            # For multi-word wake phrases, check for exact or similar sequences
                                            if len(wake_words) > 1:
                                                # Check if wake phrase is an exact substring
                                                if self.wake_word in partial_text:
                                                    found_exact = True
                                                    if self.debug:
                                                        print(f"  Exact wake phrase found in partial: '{partial_text}'")
                                                else:
                                                    # Try more flexible matching for partial results
                                                    # For two-word phrases like "hello computer"
                                                    if len(wake_words) == 2:
                                                        # More flexible first word matching
                                                        first_word_matches = [i for i, w in enumerate(words) if w == wake_words[0].lower() 
                                                            or w.startswith(wake_words[0].lower())
                                                            or Levenshtein.distance(w, wake_words[0].lower()) <= 1]
                                                        
                                                        # More flexible second word matching - especially for "computer"
                                                        if wake_words[1].lower() == "computer":
                                                            # For "computer", be extra flexible
                                                            computer_variants = ["computer", "computers", "computing", "compute", "compter", "computed"]
                                                            second_word_matches = [i for i, w in enumerate(words) if 
                                                                w in computer_variants or w.startswith("comput") or 
                                                                Levenshtein.distance(w, "computer") <= 2]
                                                        else:
                                                            # Normal flexible matching for other second words
                                                            second_word_matches = [i for i, w in enumerate(words) if 
                                                                w == wake_words[1].lower() or 
                                                                w.startswith(wake_words[1].lower()) or
                                                                Levenshtein.distance(w, wake_words[1].lower()) <= 1]
                                                        
                                                        # Check if words appear in sequence (not necessarily consecutive)
                                                        for pos1 in first_word_matches:
                                                            for pos2 in second_word_matches:
                                                                # Single if statement to check position and distance
                                                                if pos1 < pos2 and pos2 - pos1 <= 4:  # Allow up to 3 words between
                                                                    if self.debug:
                                                                        print(f"  Found wake words in sequence in partial: '{words[pos1]}...{words[pos2]}' (distance: {pos2-pos1})")
                                                                    found_exact = True
                                                                    break  # Break inner loop when match found
                                                            if found_exact:
                                                                break  # Break outer loop when match found
                                            if found_exact:
                                                # Only process if enough time has passed since last detection attempt
                                                current_time = time.time()
                                                if current_time - self.last_potential_detection_time < self.min_detection_gap:
                                                    continue
                                                
                                                if self.debug:
                                                    print(f"  Found wake word in partial result")
                                                # Force a final recognition to verify the partial result
                                                if self.recognizer.AcceptWaveform(audio_buffer):
                                                    final_result_json = self.recognizer.Result()
                                                    final_result_data = json.loads(final_result_json)
                                                    final_text = final_result_data.get('text', '').lower()
                                                    
                                                    if self.debug and final_text:
                                                        print(f"  Final verification: '{final_text}'")
                                                    
                                                    # Check if wake word is still present in final recognition
                                                    # First try exact match
                                                    exact_match = self.wake_word in final_text
                                                    if self.debug:
                                                        print(f"  🔍 Checking wake word match in: '{final_text}'")
                                                        print(f"  - Exact match: {exact_match}")
                                                    
                                                    # If no exact match, try Levenshtein distance for entire phrase
                                                    fuzzy_match = False
                                                    if not exact_match:
                                                        # Split into words for comparison
                                                        wake_words = self.wake_word.split()
                                                        final_words = final_text.split()
                                                        
                                                        # For multi-word wake phrases, check for fuzzy sequence match
                                                        if len(wake_words) > 1 and len(final_words) >= len(wake_words):
                                                            # Iterate through possible start positions in final_words
                                                            for i in range(len(final_words) - len(wake_words) + 1):
                                                                # Check if this starting position could be a match
                                                                fuzzy_sequence_match = True
                                                                match_count = 0
                                                                
                                                                # Compare each wake word against the corresponding word in final_words
                                                                for j, target_word in enumerate(wake_words):
                                                                    actual_word = final_words[i+j]
                                                                    
                                                                    # Exact match always counts
                                                                    if actual_word == target_word:
                                                                        match_count += 1
                                                                        continue
                                                                    
                                                                    # Allow more forgiving matching for longer phrases
                                                                    distance = Levenshtein.distance(actual_word, target_word)
                                                                    # More forgiving distance for wake word detection
                                                                    max_allowed_distance = 2 if len(target_word) > 3 else 1
                                                                    
                                                                    if distance > max_allowed_distance:
                                                                        fuzzy_sequence_match = False
                                                                        break
                                                                
                                                                # If this is a match, set fuzzy_match to True and break the loop
                                                                if fuzzy_sequence_match:
                                                                    # Be more lenient - require at least 1 match, or 1/3 of words for longer phrases
                                                                    if match_count >= max(1, len(wake_words) // 3):
                                                                        fuzzy_match = True
                                                                        if self.debug:
                                                                            print(f"  - Found fuzzy sequence match at position {i} (exact matches: {match_count}/{len(wake_words)})")
                                                                        break
                                                    wake_word_found = exact_match or fuzzy_match
                                                    
                                                    if self.debug:
                                                        if wake_word_found:
                                                            print(f"  ✓ WAKE WORD FOUND in final verification: {exact_match and 'exact match' or 'fuzzy match'}")
                                                        else:
                                                            print(f"  ✗ No wake word found in final verification")
                                                    
                                                    if wake_word_found:
                                                        # Update detection tracking
                                                        self.last_potential_detection_time = time.time()
                                                        # Check energy level
                                                        # Check energy level with more lenient thresholds for partial detections
                                                        energy_sufficient = False
                                                        if hasattr(self, 'smoothed_energy') and self.smoothed_energy > 0:
                                                            # More lenient threshold (25% margin) for partial matches
                                                            energy_sufficient = self.smoothed_energy >= (self.recording_energy_threshold * 0.75)
                                                        else:
                                                            # More lenient threshold (20% margin) for partial matches
                                                            energy_sufficient = self.current_energy >= (self.recording_energy_threshold * 0.8)
                                                        
                                                        # Show debug info
                                                        if self.debug:
                                                            # Setup for debug output
                                                            energy_ratio = self.current_energy / self.recording_energy_threshold
                                                            speech_ratio = self.current_energy / self.energy_threshold
                                                            smoothed_ratio = self.smoothed_energy / self.recording_energy_threshold if hasattr(self, 'smoothed_energy') else 0
                                                            threshold_status = "✓" if energy_sufficient else "✗"
                                                            
                                                            if energy_sufficient:
                                                                print(f"  Energy: {self.current_energy:.6f} ✓ (Rec threshold: {self.recording_energy_threshold:.6f}, Ratio: {energy_ratio:.2f}x)")
                                                                print(f"  Speech threshold: {self.energy_threshold:.6f}, Ratio: {speech_ratio:.2f}x")
                                                                if hasattr(self, 'smoothed_energy'):
                                                                    print(f"  Smoothed: {self.smoothed_energy:.6f} ✓ (Ratio: {smoothed_ratio:.2f}x)")
                                                            else:
                                                                print(f"  Energy: {self.current_energy:.6f} ✗ (Rec threshold: {self.recording_energy_threshold:.6f}, Ratio: {energy_ratio:.2f}x)")
                                                                print(f"  Speech threshold: {self.energy_threshold:.6f}, Ratio: {speech_ratio:.2f}x")
                                                                if hasattr(self, 'smoothed_energy'):
                                                                    print(f"  Smoothed: {self.smoothed_energy:.6f} ✗ (Ratio: {smoothed_ratio:.2f}x)")
                                                        if energy_sufficient:
                                                            print(f"   Starting recording...")
                                                            # Reset counter
                                                            self.consecutive_detection_count = 0
                                                            self.last_detection_time = time.time()
                                                            try:
                                                                record_thread = Thread(target=self._record_audio)
                                                                record_thread.start()
                                                                record_thread.join()
                                                            except Exception as e:
                                                                self.logger.error(f"Error in recording thread: {e}", exc_info=True)
                                                            finally:
                                                                audio_buffer = b''
                                                                continue
                                                        elif self.debug:
                                                            print(f"  ⚠️ Wake word detected in partial but energy too low for recording")
                                except Exception as e:
                                    self.logger.error(f"Error processing partial result: {e}", exc_info=True)
                                    
                            # Maintain buffer
                            # At least 200ms of audio (6400 bytes at 16kHz, 16-bit mono)
                            min_overlap = 6400
                            # Use the larger of min_overlap or buffer_overlap*2.5 for better overlap
                            keep_size = max(min_overlap, int(self.buffer_overlap * 2.5))
                            
                            # Trim buffer while keeping enough overlap
                            old_size = len(audio_buffer)
                            new_size = min(old_size, keep_size)
                            
                            if self.debug:
                                print(f"  📦 Trimming buffer from {old_size} to {new_size} bytes ({new_size/self.sample_rate/2:.2f}s)")
                                
                            audio_buffer = audio_buffer[-new_size:] if new_size > 0 else b''
                    else:
                        # Count silence frames
                        speechless_frames += 1
                        # If enough silence, reset buffer
                        if speechless_frames > max_speechless_frames and audio_buffer:
                            # One last check for wake word before resetting buffer
                            if len(audio_buffer) > self.recognition_frame_size / 2:
                                try:
                                    if self.recognizer.AcceptWaveform(audio_buffer):
                                        reset_result_json = self.recognizer.Result()
                                        reset_result_data = json.loads(reset_result_json)
                                        reset_text = reset_result_data.get('text', '').lower()
                                        
                                        if reset_text and self.debug:
                                            print(f"\n🔍 Final before reset: '{reset_text}'")
                                        
                                        # Simple check for wake word in reset buffer
                                        if reset_text and (self.wake_word in reset_text or self._find_wake_word_match(reset_text, reset_result_data)):
                                            
                                            print(f"\n⚡ Wake word detected at silence reset: '{reset_text}'")
                                            
                                            if self.debug:
                                                print(f"  Wake word found in reset buffer: '{reset_text}'")
                                            
                                            # Check energy level - simpler energy check with more forgiveness for silence reset
                                            # Check energy level - much more forgiving for silence reset detections
                                            energy_sufficient = False
                                            if hasattr(self, 'smoothed_energy') and self.smoothed_energy > 0:
                                                # Very lenient threshold (30% margin) for silence reset detections
                                                energy_sufficient = self.smoothed_energy >= (self.recording_energy_threshold * 0.7)
                                            elif hasattr(self, 'current_energy'):
                                                # Very lenient threshold (25% margin) for silence reset detections
                                                energy_sufficient = self.current_energy >= (self.recording_energy_threshold * 0.75)
                                            if self.debug:
                                                energy_ratio = self.current_energy / self.recording_energy_threshold
                                                smoothed_ratio = self.smoothed_energy / self.recording_energy_threshold if hasattr(self, 'smoothed_energy') else 0
                                                
                                                if energy_sufficient:
                                                    print(f"  Energy level: {self.current_energy:.6f} ✓ (Threshold: {self.recording_energy_threshold:.6f}, Ratio: {energy_ratio:.2f}x)")
                                                    if hasattr(self, 'smoothed_energy'):
                                                        print(f"  Smoothed: {self.smoothed_energy:.6f} ✓ (Ratio: {smoothed_ratio:.2f}x)")
                                                else:
                                                    print(f"  Energy level: {self.current_energy:.6f} ✗ (Threshold: {self.recording_energy_threshold:.6f}, Ratio: {energy_ratio:.2f}x)")
                                                    if hasattr(self, 'smoothed_energy'):
                                                        print(f"  Smoothed: {self.smoothed_energy:.6f} ✗ (Ratio: {smoothed_ratio:.2f}x)")
                                            # Start recording if energy sufficient
                                            if energy_sufficient:
                                                print(f"   Starting recording...")
                                                self.last_detection_time = time.time()
                                                try:
                                                    record_thread = Thread(target=self._record_audio)
                                                    record_thread.start()
                                                    record_thread.join()
                                                except Exception as e:
                                                    self.logger.error(f"Error in recording thread: {e}", exc_info=True)
                                                finally:
                                                    audio_buffer = b''
                                                    continue
                                            elif self.debug:
                                                print(f"  ⚠️ Wake word detected at reset but energy too low")
                                except Exception as e:
                                    self.logger.error(f"Error processing reset buffer: {e}", exc_info=True)
                            
                            # Reset buffer after silence
                            audio_buffer = b''
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.005)
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self._cleanup()
            print("Shutdown complete.")

def main():
    """Parse arguments and run the wake word recorder"""
    parser = argparse.ArgumentParser(description='Local Wake Word Detector and Audio Recorder')
    
    parser.add_argument('--model', default='model',
                        help='Path to Vosk model directory (default: model)')
    parser.add_argument('--wake_word', default='hello computer',
                        help='Wake word or phrase to listen for (default: "hello computer")')
    parser.add_argument('--audio_dir', default='recordings',
                        help='Directory to save recordings (default: recordings)')
    parser.add_argument('--vad_mode', type=int, choices=[0, 1, 2, 3], default=3,
                        help='VAD aggressiveness (0-3, default: 3)')
    parser.add_argument('--silence_duration', type=float, default=2.0,
                        help='Silence duration to stop recording in seconds (default: 2.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
    recorder = LocalWakeWordRecorder(
        model_path=args.model,
        wake_word=args.wake_word,
        audio_dir=args.audio_dir,
        vad_mode=args.vad_mode,
        silence_duration=args.silence_duration,
        debug=args.debug
    )
    
    recorder.run()

if __name__ == '__main__':
    main()

