import os

from logger import get_logger

logger = get_logger()


class Symbols(object):
    def __init__(self, label_context, filename=None):
        self.label_context = label_context
        self.blank = '<blank>'
        self.padding = '<padding>'
        self.counter = 0
        self.filename = filename
        self.sym_to_id = {}
        self.id_2_sym = {}
        if filename and os.path.exists(filename):
            logger.info('Reading output symbols from: ' + filename)
            with open(filename, 'r') as f:
                for line in f:
                    items = line.strip().split(' ')
                    self.sym_to_id[items[0]] = int(items[1])
                    if self.counter < int(items[1]):
                        self.counter = int(items[1])
            self.counter += 1
            self.id_2_sym = dict([[v, k] for k, v in self.sym_to_id.items()])

    def insert_sym(self, sym):
        if not sym in self.sym_to_id:
            self.sym_to_id[sym] = self.counter
            self.id_2_sym[self.counter] = sym
            self.counter += 1
        return self.sym_to_id[sym]

    def insert_blank(self):
        return self.insert_sym(self.blank)

    def insert_padding(self):
        return self.insert_sym(self.padding)

    def get_padding_id(self):
        return self.sym_to_id[self.padding]

    def get_id(self, sym):
        return self.sym_to_id[sym]

    def get_sym(self, id):
        return self.id_2_sym[id]

    def get_all_ids(self, id):
        return list(self.sym_to_id.values())

    def convert_to_str(self, l):
        if self.label_context > 0:
            str_decoded = ''.join([self.get_sym(x)[self.label_context:-self.label_context] for x in l])
        else:
            str_decoded = ''.join([self.get_sym(x) for x in l])
        str_decoded = str_decoded.replace(self.blank, '')
        str_decoded = str_decoded.replace('_', ' ')
        return str_decoded

    def write(self, filename=None):
        if not filename:
            filename = self.filename
        logger.info('Writing output symbols to: ' + filename)
        with open(filename, 'w') as f:
            for k, v in sorted(self.sym_to_id.items()):
                f.write(k + " " + str(v) + "\n")
