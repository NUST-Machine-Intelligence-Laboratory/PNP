# -*- coding: utf-8 -*-
import os
from datetime import datetime
from easydict import EasyDict

# ---------- Color Printing ----------
PStyle = EasyDict({
    'end': '\33[0m',
    'bold': '\33[1m',
    'italic': '\33[3m',
    'underline': '\33[4m',
    'selected': '\33[7m',
    'red': '\33[31m',
    'green': '\33[32m',
    'yellow': '\33[33m',
    'blue': '\33[34m'
})


# ---------- Naive Print Tools ----------
def print_to_logfile(logfile, content, init=False, end='\n'):
    if init:
        with open(logfile, 'w') as f:
            f.write(content + end)
    else:
        with open(logfile, 'a') as f:
            f.write(content + end)


def print_to_console(content, style=None, color=None):
    flag = 0
    if color in PStyle.keys():
        content = f'{PStyle[color]}{content}'
        flag += 1
    if style in PStyle.keys():
        content = f'{PStyle[style]}{content}'
        flag += 1
    if flag > 0:
        content = f'{content}{PStyle.end}'
    print(content, flush=True)


# ---------- Simple Logger ----------
class Logger(object):
    def __init__(self, logging_dir, DEBUG=False):
        # set up logging directory
        self.DEBUG = DEBUG
        self.logging_dir = logging_dir
        self.logfile_path = None
        self.debug_info_path = None
        os.makedirs(self.logging_dir, exist_ok=True)

    def set_logfile(self, logfile_name):
        f = open(f'{self.logging_dir}/{logfile_name}', 'w')
        f.close()
        f = open(f'{self.logging_dir}/debug-{logfile_name}', 'w')
        f.close()
        f = open(f'{self.logging_dir}/msg-{logfile_name}', 'w')
        self.logfile_path = f'{self.logging_dir}/{logfile_name}'
        self.debug_info_path = f'{self.logging_dir}/debug-{logfile_name}'
        self.msg_info_path = f'{self.logging_dir}/msg-{logfile_name}'

    def debug(self, content):
        if self.DEBUG:
            assert self.debug_info_path is not None
            print_to_logfile(logfile=self.debug_info_path, content=content, init=False)
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.yellow}DEBUG{PStyle.end}    | - {PStyle.yellow}{content}{PStyle.end}')

    def info(self, content):
        assert self.logfile_path is not None
        print_to_logfile(logfile=self.logfile_path, content=content, init=False)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.blue}INFO{PStyle.end}     | - {PStyle.blue}{content}{PStyle.end}')

    def msg(self, content):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.yellow}MSG{PStyle.end}    | - {PStyle.yellow}{content}{PStyle.end}')
        assert self.msg_info_path is not None
        print_to_logfile(logfile=self.msg_info_path, content=content, init=False)
