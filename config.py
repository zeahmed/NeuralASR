
import argparse
from configparser import ConfigParser


class Config(object):
    def __init__(self, configfile, isTest=False):
        self.cfg = ConfigParser()
        self.cfg.read(configfile)
        parameters = self.cfg['Parameters']
        self.samplerate = int(parameters['samplerate'])
        self.numcep = int(parameters['numcep'])
        self.batch_size = int(parameters['batch_size'])
        self.epochs = int(parameters['epochs'])
        self.learningrate = float(parameters['learningrate'])
        self.model_dir = parameters['model_dir']
        self.test_input = self.train_input = None
        if not isTest:
            self.train_input = parameters['train_input']
        if 'test_input' in parameters:
            self.test_input = parameters['test_input']

    def print_config(self):
        print('samplerate=', self.samplerate)
        print('numcep=', self.numcep)
        print('batch_size=', self.batch_size)
        print('epochs=', self.epochs)
        print('numclearningrateep=', self.learningrate)
        print('model_dir=', self.model_dir)
        print('train_input=', self.train_input)
        print('test_input=', self.test_input)

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
