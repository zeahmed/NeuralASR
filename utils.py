import re

import numpy as np

import librosa
from python_speech_features import mfcc


def include_context(audio_mfcc, numcontext, numcep):
    num_strides = len(audio_mfcc)
    empty_context = np.zeros((numcontext, numcep), dtype=audio_mfcc.dtype)
    audio_mfcc = np.concatenate((empty_context, audio_mfcc, empty_context))

    window_size = 2 * numcontext + 1
    audio_mfcc = np.lib.stride_tricks.as_strided(
        audio_mfcc,
        (num_strides, window_size, numcep),
        (audio_mfcc.strides[0], audio_mfcc.strides[0], audio_mfcc.strides[1]),
        writeable=False)

    audio_mfcc = np.reshape(audio_mfcc, [num_strides, -1])
    return np.copy(audio_mfcc)


def convert_to_mfcc(wavfile, sr, numcontext, numcep):
    audio, _ = librosa.load(wavfile, mono=True, sr=sr)
    audio_mfcc = mfcc(audio, samplerate=sr, numcep=numcep)
    if numcontext > 0:
        audio_mfcc = include_context(audio_mfcc, numcontext, numcep)
    audio_mfcc = (audio_mfcc - np.mean(audio_mfcc)) / np.std(audio_mfcc)
    return audio_mfcc.astype(np.float32)


def read_label_text(txtfile, punc_regex):
    with open(txtfile, 'r') as f:
        transcription = f.read().replace('\n', '').replace('\r', '')

    transcription = transcription.strip().lower()
    clean_transcription = re.sub(punc_regex, '', transcription)
    clean_transcription = clean_transcription.replace('  ', ' ').replace(' ', '_')
    return clean_transcription


def compute_mfcc_and_read_transcription(wavfile, sr, numcontext, numcep, punc_regex=None, txtfile=None):
    audio_mfcc = convert_to_mfcc(wavfile, sr, numcontext, numcep)

    if txtfile:
        clean_transcription = read_label_text(txtfile, punc_regex)
        return audio_mfcc, clean_transcription
    else:
        return audio_mfcc
