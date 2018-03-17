

class AudioSample(object):
    '''
    A class to hold training instances.
    '''

    def __init__(self, id, mfcc, labels, transcription):
        self.id = id
        self.mfcc = mfcc
        self.labels = labels
        self.transcription = transcription
