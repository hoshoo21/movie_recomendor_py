from transformers import pipeline

pipe = pipeline(
    "text-to-speech",
    model="TalhaAhmed/Urdu_kaani_TTS"
)

text = "ایک دن ایک بوڑھا آدمی بازار گیا اور اس نے کہا کہ آج موسم بہت خوشگوار ہے۔"

audio = pipe(text)

with open("output.wav", "wb") as f:
    f.write(audio["audio"])