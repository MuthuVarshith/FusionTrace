
import torch
import torchaudio
import os
import uuid
from pydub import AudioSegment
import shutil
import soundfile as sf
import tempfile
from transformers.models.wav2vec2  import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from .config import logger, TEST_DATA_DIR


def check_ffmpeg():
    """
    Check if ffmpeg is available.
    """
    return shutil.which("ffmpeg") is not None

def validate_wav_file(wav_path):
    """
    Validate that the WAV file is readable and has expected properties.
    """
    try:
        # Check file size
        file_size = os.path.getsize(wav_path)
        if file_size < 44:  # Minimum WAV header size
            logger.error(f"WAV file {wav_path} is too small: {file_size} bytes")
            return False
        
        # Read with soundfile
        data, sr = sf.read(wav_path)
        logger.info(f"Validated WAV file: {wav_path}, Sample rate: {sr}, Shape: {data.shape}")
        
        # Check sample rate and duration
        expected_samples = 16000 * 4  # 4 seconds at 16kHz
        if sr != 16000:
            logger.warning(f"Unexpected sample rate: {sr}, expected 16000")
        if len(data) > expected_samples:
            logger.warning(f"Unexpected duration: {len(data)/sr}s, expected <= 4s")
        
        return True
    except Exception as e:
        logger.error(f"Failed to validate WAV file {wav_path}: {str(e)}")
        return False

def preprocess_audio(audio_path, sample_rate=16000, max_length=4.0):
    """
    Preprocess audio file: convert to wav, resample, convert to mono, normalize, and trim/pad to max_length.
    Save the converted 4-second wav file in test_data directory using soundfile.
    """
    try:
        # Generate unique filename for saved wav
        unique_filename = f"converted_{uuid.uuid4().hex}.wav"
        output_wav_path = os.path.join(TEST_DATA_DIR, unique_filename)
        
        # Handle wav files directly
        if audio_path.lower().endswith('.wav'):
            logger.info(f"Processing WAV file directly: {audio_path}")
            waveform, orig_sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if orig_sample_rate != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sample_rate, sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim to 4 seconds
            max_samples = int(max_length * sample_rate)
            if waveform.shape[1] > max_samples:
                logger.info("Trimming WAV to 4 seconds")
                waveform = waveform[:, :max_samples]
            
            # Convert to numpy for soundfile
            waveform_np = waveform.squeeze(0).numpy()
            
            # Save to test_data using soundfile
            logger.info(f"Saving WAV to: {output_wav_path}")
            sf.write(output_wav_path, waveform_np, sample_rate)
            
        # Handle mp3 files with pydub
        elif audio_path.lower().endswith('.mp3'):
            if not check_ffmpeg():
                raise RuntimeError("FFmpeg not found. Required for MP3 processing.")
            logger.info(f"Processing MP3 file: {audio_path}")
            audio = AudioSegment.from_mp3(audio_path)
            
            # Trim to 4 seconds
            max_length_ms = max_length * 1000
            if len(audio) > max_length_ms:
                logger.info("Trimming MP3 to 4 seconds")
                audio = audio[:max_length_ms]
            
            # Set to mono and 16kHz
            audio = audio.set_frame_rate(sample_rate).set_channels(1)
            
            # Save to temporary file
            temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            logger.info(f"Saving temporary WAV to: {temp_wav_path}")
            audio.export(temp_wav_path, format="wav")
            
            # Validate temporary file
            if not validate_wav_file(temp_wav_path):
                os.remove(temp_wav_path)
                raise RuntimeError(f"Invalid temporary WAV file: {temp_wav_path}")
            
            # Move to test_data
            logger.info(f"Moving WAV to: {output_wav_path}")
            shutil.move(temp_wav_path, output_wav_path)
            
            # Load with torchaudio
            waveform, orig_sample_rate = torchaudio.load(output_wav_path)
        
        else:
            raise ValueError("Unsupported file format. Only .wav and .mp3 are supported.")
        
        # Validate the final WAV file
        if not validate_wav_file(output_wav_path):
            raise RuntimeError(f"Invalid WAV file created: {output_wav_path}")
        
        # Normalize (zero mean, unit variance)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        
        # Trim or pad to 4 seconds (redundant but ensures consistency)
        max_samples = int(max_length * sample_rate)
        num_samples = waveform.shape[1]
        if num_samples > max_samples:
            waveform = waveform[:, :max_samples]
        elif num_samples < max_samples:
            padding = torch.zeros(1, max_samples - num_samples)
            waveform = torch.cat([waveform, padding], dim=1)
        
        return waveform.squeeze(0), output_wav_path
    
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        raise

def predict_audio(audio_path, model, processor):
    """
    Predict whether the audio is real or fake and return the label, confidence score, and saved file path.
    """
    try:
        # Preprocess audio (saves to test_data)
        waveform, saved_wav_path = preprocess_audio(audio_path)
        
        # Ensure waveform is a 1D tensor or numpy array for processor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # Process waveform with Wav2Vec2Processor
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label = logits.argmax(dim=-1).item()
            confidence = probabilities[0, predicted_label].item()
        
        # Convert to class name and confidence percentage
        class_name = "Real" if predicted_label == 0 else "Fake"
        confidence_percent = confidence * 100
        
        logger.info(f"Prediction: {class_name}, Confidence: {confidence_percent:.2f}%, Saved to: {saved_wav_path}")
        return class_name, f"{confidence_percent:.2f}%", saved_wav_path
    
    except Exception as e:
        logger.error(f"Error in predict_audio: {str(e)}")
        raise