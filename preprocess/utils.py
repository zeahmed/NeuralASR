import re

import numpy as np

import librosa
from python_speech_features import mfcc


def convert_to_mfcc(wavfile, sr, numcep):
    audio, _ = librosa.load(wavfile, mono=True, sr=sr)
    audio_mfcc = mfcc(audio, samplerate=sr, numcep=numcep)
    #audio_mfcc = np.expand_dims(audio_mfcc, axis=0)
    audio_mfcc = (audio_mfcc - np.mean(audio_mfcc)) / np.std(audio_mfcc)
    return audio_mfcc


def get_labels(txtfile):
    with open(txtfile, 'r') as f:
        transcription = f.read().replace('\n', '').replace('\r', '')

    transcription = transcription.strip().lower()
    clean_transcription = re.sub('[^a-z0-9\s]', '', transcription)
    clean_transcription = clean_transcription.replace('  ', ' ')
    return clean_transcription


def convert_inputs_to_ctc_format(wavfile, sr, numcep, txtfile=None):
    audio_mfcc = convert_to_mfcc(wavfile, sr, numcep)
    seq_len = np.asarray(audio_mfcc.shape[0], dtype=np.int32)

    if txtfile:
        clean_transcription = get_labels(txtfile)
        return audio_mfcc.astype(np.float32), seq_len, clean_transcription
    else:
        return audio_mfcc.astype(np.float32), seq_len
