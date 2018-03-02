import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from .utils import convert_inputs_to_ctc_format 

class SpeechSample(object):
    def __init__(self, id, mfcc, target):
        self.id = id
        self.mfcc = mfcc
        self.target = target
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert audio files into mfcc for training ASR")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input csv file. 1st column is audio file path and 2nd column is transcription file path. Third column is size of audio file (needed for sorting).")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to directory where .pkl file will be generated.")
    parser.add_argument("-sr", "--samplerate", required=False, default=8000, help="Audio sample rate to use for conversion and loading audio file.")
    parser.add_argument("-n", "--numcep", required=False, default=13, help="Number of cepstral coefficients")
    args = parser.parse_args()
    
    data = pd.read_csv(args.input, header = None, sep=',')
    data.sort_values(by=2, inplace=True)
    train_X = data.ix[:, 0].values.ravel()
    train_Y = data.ix[:, 1].values.ravel()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    np.set_printoptions(suppress=True)
    scp_file = os.path.join(args.output_dir, "train.scp")
    with open(scp_file, 'w') as f:
        for i in range(len(train_X)):
            if os.path.exists(train_X[i]) and os.path.exists(train_Y[i]):
                print(train_X[i])
                mfcc, target, seq_len, clean_transcription = convert_inputs_to_ctc_format(train_X[i], args.samplerate, args.numcep, train_Y[i])
                filename = os.path.basename(train_X[i]).replace(".wav","")
                filepath = os.path.join(args.output_dir, filename + ".pkl")
                f.write(filepath+",\""+clean_transcription+"\"\n")
                with open(filepath, 'wb') as output:
                    speechsample = SpeechSample(filename,
                                                mfcc,
                                                target)
                    pickle.dump(speechsample, output, pickle.HIGHEST_PROTOCOL)
                