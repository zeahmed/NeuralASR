import argparse
import os
from configparser import ConfigParser


class Config(object):
    def __init__(self, configfile, isTest=False):
        self.cfg = ConfigParser()
        self.cfg.read(configfile)
        parameters = self.cfg['Parameters']
        self.samplerate = int(parameters['samplerate'])
        self.numcep = int(parameters['numcep'])
        self.numcontext = int(parameters['numcontext']
                              ) if 'numcontext' in parameters else 0
        self.feature_size = (2 * self.numcontext + 1) * self.numcep
        self.batch_size = int(parameters['batch_size'])
        self.epochs = int(parameters['epochs'])
        self.learningrate = float(parameters['learningrate'])
        self.model_dir = parameters['model_dir']
        self.punc_regex = parameters['punc_regex']
        self.train_input = None
        self.test_input = None
        self.mfcc_input = None
        self.mfcc_output = None
        self.sym_file = None

        if 'sym_file' in parameters:
            self.sym_file = parameters['sym_file']

        parameters = self.cfg['Train']
        if 'input' in parameters:
            self.train_input = parameters['input']

        parameters = self.cfg['MFCC Featurizer']
        if 'input' in parameters:
            self.mfcc_input = parameters['input']
        if 'output' in parameters:
            self.mfcc_output = parameters['output']

        parameters = self.cfg['Test']
        if 'input' in parameters:
            self.test_input = parameters['input']
        else:
            raise ValueError(
                "Missing 'test_input' in configuration file: " + configfile)

    def print_config(self):
        print('samplerate=', self.samplerate)
        print('numcep=', self.numcep)
        print('numcontext=', self.numcontext)
        print('batch_size=', self.batch_size)
        print('epochs=', self.epochs)
        print('numclearningrateep=', self.learningrate)
        print('model_dir=', self.model_dir)
        print('punc_regex=', self.punc_regex)
        print('sym_file=', self.sym_file)
        print('train_input=', self.train_input)
        print('test_input=', self.test_input)
        print('mfcc_input=', self.mfcc_input)
        print('mfcc_output=', self.mfcc_output)

    def write(self, filename):
        with open(filename, 'w') as configfile:
            self.cfg.write(configfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    parser.add_argument("-t", "--isTest", required=False, default=False,
                        help="Is configuration file for testing?")
    args = parser.parse_args()

    config = Config(args.config, args.isTest)
    config.print_config()
