import argparse
import importlib
import os
from configparser import ConfigParser, ExtendedInterpolation

from logger import get_logger
from symbols import Symbols

logger = get_logger()

class Config(object):
    def __init__(self, configfile, isTraining=False):
        self.isTraining = isTraining
        self.configfile = configfile
        logger.info('Reading configuration from: ' + configfile)
        self.cfg = ConfigParser(interpolation=ExtendedInterpolation())
        self.cfg.read(configfile)
        parameters = self.cfg['Parameters']
        self.samplerate = int(parameters['samplerate'])
        self.numcep = int(parameters['numcep'])
        self.numcontext = int(parameters['numcontext']
                              ) if 'numcontext' in parameters else 0
        self.rand_shift = int(
            parameters['rand_shift']) if 'rand_shift' in parameters else 0
        self.feature_size = (2 * self.numcontext + 1) * self.numcep
        self.batch_size = int(parameters['batch_size'])
        self.epochs = int(parameters['epochs'])
        self.learningrate = float(parameters['learningrate'])
        self.model_dir = parameters['model_dir']
        self.start_step = int(parameters['start_step'])
        self.report_step = int(parameters['report_step'])
        self.num_gpus = int(parameters['num_gpus'])
        self.label_context = int(parameters['label_context'])

        self.batch_size = self.batch_size * \
            (self.num_gpus if self.num_gpus > 0 else 1)

        self.punc_regex = parameters['punc_regex']
        self.network = parameters['network']
        self.train_input = None
        self.test_input = None
        self.mfcc_input = None
        self.mfcc_output = None
        self.sym_file = None

        if 'sym_file' in parameters:
            self.sym_file = parameters['sym_file']

        if isTraining:
            self.symbols = Symbols(self.label_context, self.sym_file)
        else:
            self.symbols = Symbols(self.label_context)

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
        elif not self.train_input:
            raise ValueError(
                "Missing 'test_input' in configuration file: " + configfile)
    
    def load_network(self,fortraining=False):
        package = self.network.split('.')
        classname = package[-1]
        module = importlib.import_module('.'.join(package[:-1]))
        return getattr(module, classname)(self, fortraining=fortraining)

    def print_config(self):
        config_str = '\n'
        config_str += ('samplerate=%d\n' % self.samplerate)
        config_str += ('numcep=%d\n' % self.numcep)
        config_str += ('numcontext=%d\n' % self.numcontext)
        config_str += ('rand_shift=%d\n' % self.rand_shift)
        config_str += ('batch_size=%d\n' % self.batch_size)
        config_str += ('epochs=%d\n' % self.epochs)
        config_str += ('learningrate=%f\n' % self.learningrate)
        config_str += ('model_dir=%s\n' % self.model_dir)
        config_str += ('start_step=%d\n' % self.start_step)
        config_str += ('report_step=%d\n' % self.report_step)
        config_str += ('num_gpus=%d\n' % self.num_gpus)
        config_str += ('label_context=%d\n' % self.label_context)
        config_str += ('punc_regex=%s\n' % self.punc_regex)
        config_str += ('network=%s\n' % self.network)
        config_str += ('sym_file=%s\n' % self.sym_file)
        config_str += ('train_input=%s\n' % self.train_input)
        config_str += ('test_input=%s\n' % self.test_input)
        config_str += ('mfcc_input=%s\n' % self.mfcc_input)
        config_str += ('mfcc_output=%s\n' % self.mfcc_output)
        logger.info(config_str)

    def write_symbols(self):
        self.symbols.write(self.sym_file)

    def write(self, filename):
        logger.info('Writing configuration to: ' + filename)
        with open(filename, 'w') as configfile:
            self.cfg.write(configfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("config", help="Configuration file.")
    parser.add_argument("-t", "--isTraining", required=False, default=False,
                        help="Is configuration file for testing?")
    args = parser.parse_args()

    config = Config(args.config, args.isTraining)
    config.print_config()
