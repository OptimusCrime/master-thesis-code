#!/usr/bin/env python
# -*- coding: utf-8 -*-

class LogPrettifier:

    FIRST_EPOCH = -1
    EPOCH = 0
    INFO = 1
    END = 2

    LEFT = 0
    CENTER = 1

    def __init__(self, log):
        self.log = log

    def write(self, line, line_type=INFO):
        # Opening/continuing line
        if line_type == LogPrettifier.FIRST_EPOCH:
            self.log.info('╔══════════════════════════════════════════════════════════╗')
        if line_type == LogPrettifier.EPOCH:
            self.log.info('╠══════════════════════════════════════════════════════════╣')
        if line_type == LogPrettifier.END:
            self.log.info('╚══════════════════════════════════════════════════════════╝')

        # Spacing
        if line_type == LogPrettifier.FIRST_EPOCH or line_type == LogPrettifier.EPOCH:
            self.log.info('║                                                          ║')

        # Actual line content
        self.log.info(LogPrettifier.construct_output(line, line_type))

        # More spacing
        if line_type == LogPrettifier.FIRST_EPOCH or line_type == LogPrettifier.EPOCH:
            self.log.info('║                                                          ║')
            self.log.info('╠══════════════════════════════════════════════════════════╣')

    @staticmethod
    def construct_output(line, line_type):
        alignment = LogPrettifier.CENTER
        if line_type == LogPrettifier.INFO:
            alignment = LogPrettifier.LEFT
        pad_left, pad_right = LogPrettifier.calculate_padding(line, alignment)

        output_line  = '║'
        output_line += LogPrettifier.repeat_to_length(' ', pad_left)
        output_line += line
        output_line += LogPrettifier.repeat_to_length(' ', pad_right)
        output_line += '║'

        return output_line

    @staticmethod
    def calculate_padding(line, alignment):
        if alignment == LogPrettifier.CENTER:
            return LogPrettifier.calculate_padding_center(line)

        return LogPrettifier.calculate_padding_left(line)

    @staticmethod
    def calculate_padding_center(line):
        pad_length = 58 - len(line)
        if pad_length % 2 == 0:
            return pad_length // 2, pad_length // 2
        return pad_length // 2, (pad_length // 2) + 1

    @staticmethod
    def calculate_padding_left(line):
        return 1, 57 - len(line)

    @staticmethod
    def repeat_to_length(string_to_expand, length):
        return (string_to_expand * int((length / len(string_to_expand)) + 1))[:length]
