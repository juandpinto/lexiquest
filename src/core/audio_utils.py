import whisper
import numpy as np
import sounddevice as sd

from TTS.api import TTS


def get_stt_model(model_name: str = 'tiny'):
    return whisper.load_model(model_name)


def get_tts_model(model_name: str = 'tts_models/multilingual/multi-dataset/xtts_v2'):
    return TTS(model_name=model_name, progress_bar=False, gpu=True)


def speak_text(tts_model: TTS, text: str, using_voice_clone=False):
    if using_voice_clone:
        wav = tts_model.tts(text, language="en", speaker_wav="../harvard.wav")
    else:
        wav = tts_model.tts(text)

    wav = np.array(wav, dtype=np.float32)
    sd.play(wav, samplerate=48050)  # Sample rate should match model (22050 is common)
    sd.wait()


def transcribe_speech(stt_model, file_location: str):
    transcription = stt_model.transcribe(file_location)
    return transcription['text']


if __name__ == "__main__":
    tts = get_tts_model("tts_models/en/jenny/jenny")
    text_to_speak = 'Greetings earthling! You have been chosen to join the Earth Space Command.' \
                    ' Youre mission is to discover whether there is intelligent life in nearby planets and stars.' \
                    ' If you accept this mission, you must make decisions based on the choices that arise before you.' \
                    ' Some challenges will test your wits, others will be like stealing candy from a baby alien, ' \
                    ' but all will bring glory to your species. Do you accept your mission?'
    speak_text(tts, text_to_speak)
    # model = whisper.load_model("tiny")
    # result = model.transcribe("../../whisper_test_fixed.mp3")
    # print(result["text"])
