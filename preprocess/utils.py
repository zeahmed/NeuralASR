import numpy as np
import re
import librosa
from python_speech_features import mfcc

START_INDEX = ord('a') - 1  # 0 is reserved for space

def convert_to_mfcc(wavfile, sr, numcep):
    audio, _ = librosa.load(wavfile, mono=True, sr=sr)
    audio_mfcc = mfcc(audio, samplerate=sr, numcep=numcep)
    audio_mfcc = np.expand_dims(audio_mfcc, axis=0)
    audio_mfcc = (audio_mfcc - np.mean(audio_mfcc)) / np.std(audio_mfcc)
    return audio_mfcc

def get_labels(txtfile):
    with open(txtfile, 'r') as f:
        transcription = f.read().replace('\n', '').replace('\r', '')

    transcription = transcription.strip().lower()
    clean_transcription = re.sub('[^a-z0-9\s]', '', transcription)
    clean_transcription = clean_transcription.replace('  ', ' ')
    targets = np.asarray([0 if x == ' ' else ord(x) - START_INDEX for x in clean_transcription])
    targets = sparse_tuple_from([targets])
    return targets, clean_transcription

def convert_inputs_to_ctc_format(wavfile, sr, numcep, txtfile):
    audio_mfcc = convert_to_mfcc(wavfile, sr, numcep)
    seq_len = [audio_mfcc.shape[1]]
    targets, clean_transcription = get_labels(txtfile)
    return audio_mfcc, targets, seq_len, clean_transcription


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths