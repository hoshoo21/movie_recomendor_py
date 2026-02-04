from datasets import load_dataset
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
# Load dataset
dataset = load_dataset("humairawan/Urdu-LjSpeech")
# Initialize model and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")