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
    audio, _ = librosa.load(wavfile, mono=True)
    audio_mfcc = mfcc(audio, samplerate=sr, numcep=numcep)
    if numcontext > 0:
        audio_mfcc = include_context(audio_mfcc, numcontext, numcep)
    audio_mfcc = (audio_mfcc - np.mean(audio_mfcc)) / np.std(audio_mfcc)
    return audio_mfcc


def get_labels(txtfile):
    with open(txtfile, 'r') as f:
        transcription = f.read().replace('\n', '').replace('\r', '')

    transcription = transcription.strip().lower()
    clean_transcription = re.sub('[^a-z0-9\s]', '', transcription)
    clean_transcription = clean_transcription.replace('  ', ' ')
    return clean_transcription


def convert_inputs_to_ctc_format(wavfile, sr, numcontext, numcep, txtfile=None):
    audio_mfcc = convert_to_mfcc(wavfile, sr, numcontext, numcep)
    seq_len = np.asarray(audio_mfcc.shape[0], dtype=np.int32)

    if txtfile:
        clean_transcription = get_labels(txtfile)
        return audio_mfcc.astype(np.float32), seq_len, clean_transcription
    else:
        return audio_mfcc.astype(np.float32), seq_len
