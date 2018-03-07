

class AudioSample(object):
    '''
    A class to hold training instances.
    '''

    def __init__(self, id, mfcc, seq_len, transcription):
        self.id = id
        self.mfcc = mfcc
        self.seq_len = seq_len
        self.transcription = transcription
