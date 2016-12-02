#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rorschach.preprocessing.handlers import BaseHandler


class EmbeddingHandler(BaseHandler):

    def __init__(self):
        super().__init__()

        self.letter_embedded = {}

    def run(self):
        self.parse_embedding()
        self.contract()

    def parse_embedding(self):
        for set_list_obj in self.set_list:
            for content in set_list_obj['data']:
                self.create_embedding(content, set_list_obj['identifier'])

    def create_embedding(self, obj, identifier):
        expression = []
        current_offset = 0
        current_type = None
        current_length = 0

        matrix = obj['matrix'][0]

        for i in range(len(matrix)):
            if current_type is None:
                current_type = matrix[current_offset]
                current_offset += 1
                current_length = 1
                continue

            if matrix[current_offset] != current_type:
                expression.append(str(current_length) + ('B' if current_type == 0 else 'W'))
                current_length = 0
                current_type = matrix[current_offset]

            current_length += 1
            current_offset += 1

        expression.append(str(current_length) + ('B' if current_type == 0 else 'W'))

        if identifier != BaseHandler.DATA_SET:
            expression.append('0')

        # new_expression = expression
        # This code is used to concatinate one white and one black expression
        new_expression = []
        for i in range(1, len(expression) - 1):
            new_expression.append(
                expression[i - 1] + expression[i]
            )

        if len(new_expression) == 0:
            new_expression.append(expression[0])

        if identifier != BaseHandler.DATA_SET:
            obj['embedding_raw'] = new_expression
        else:
            self.letter_embedded[obj['text']] = new_expression

    def contract(self):
        data_set = None
        for set_list_obj in self.set_list:
            if set_list_obj['identifier'] == BaseHandler.DATA_SET:
                data_set = set_list_obj
                break

        for set_list_obj in self.set_list:
            if set_list_obj['identifier'] != BaseHandler.DATA_SET:
                for content in set_list_obj['data']:
                    self.contract_letters(content, data_set)

    def contract_letters(self, obj, data_set):
        contracted = []
        labels = []
        current_offset = 0

        for letter in obj['text']:
            letter_matrix = self.letter_embedded[letter]

            for i in range(current_offset, len(obj['embedding_raw'])):
                if EmbeddingHandler.is_sublist(obj['embedding_raw'][i:], letter_matrix):
                    contracted.append(''.join(obj['embedding_raw'][i:i + len(letter_matrix)]) + 'L')
                    current_offset += len(letter_matrix)
                    labels.append(letter)
                    break

                contracted.append(obj['embedding_raw'][i])
                labels.append(None)
                current_offset += 1

        # Add potential left over characters
        if current_offset != len(obj['embedding_raw']):
            contracted.extend(obj['embedding_raw'][current_offset:])

            # Can I has the awesome programmer that wrote this nice code?!
            labels.extend([None] * (len(obj['embedding_raw']) - current_offset))

        obj['embedding'] = contracted
        obj['labels_raw'] = labels

    @staticmethod
    def is_sublist(original, sub):
        if len(sub) > len(original):
            return False

        # I am a genious
        return original[0:len(sub)] == sub
