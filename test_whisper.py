from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Select an audio file and read it:
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = ds[0]["audio"]
print('audio_sample: ', audio_sample)

print('audio_sample array: ', audio_sample["array"].shape)

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Use the model and processor to transcribe the audio:
input_features = processor(
    audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt"
).input_features

print('input_features shape: ',input_features.shape)
# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])