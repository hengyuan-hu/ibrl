import os
import sys


class Logger:
    def __init__(self, path, mode="w", print_to_stdout=True):
        assert mode in {"w", "a"}, "unknown mode for logger %s" % mode
        if print_to_stdout:
            self.terminal = sys.stdout
        else:
            self.terminal = None
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if mode == "w" or not os.path.exists(path):
            self.log = open(path, "w")
        else:
            self.log = open(path, "a")

    def write(self, message):
        if self.terminal is not None:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # for python 3 compatibility.
        pass
