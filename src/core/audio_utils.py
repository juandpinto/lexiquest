import numpy as np
import sounddevice as sd

from TTS.api import TTS


def get_tts_model(model_name: str = 'tts_models/en/ljspeech/glow-tts'):
    return TTS(model_name=model_name, progress_bar=False, gpu=False)


def speak_text(text: str, tts_model: TTS):
    wav = tts_model.tts(text)
    wav = np.array(wav, dtype=np.float32)
    sd.play(wav, samplerate=22050)  # Sample rate should match model (22050 is common)
    sd.wait()


if __name__ == "__main__":
    tts = get_tts_model('tts_models/en/ljspeech/tacotron2-DDC')
    text_to_speak = 'Greetings earthling! You have been chosen to join the Earth Space Command.' \
                    ' You mission is to discover whether there is intelligent life in nearby planets and stars.' \
                    ' If you accept this mission, you must make decisions based on the choices that arise before you.' \
                    ' Some challenges will test your wits, others will be like stealing candy from a baby alien, but all will bring glory to your species. Do you accept your mission?'
    speak_text(text_to_speak, tts)
