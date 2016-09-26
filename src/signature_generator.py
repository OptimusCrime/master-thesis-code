#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image


class SignatureGenerator:

    LETTER_SET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Tuple with image size on format (width,height)
    SIGNATURE_SIZE = None

    def __init__(self):
        pass

    @staticmethod
    def generate():
        SignatureGenerator.calculate_signature_sizes()

        image = SignatureGenerator.concatenate_images()

        SignatureGenerator.create_signatures(image)

    @staticmethod
    def calculate_signature_sizes():
        # Just open a random image in the letters directory
        img = Image.open('dump/letters/' + SignatureGenerator.LETTER_SET[0] + '.png')

        SignatureGenerator.SIGNATURE_SIZE = (img.width, img.height)

    @staticmethod
    def concatenate_images():
        image = Image.new('1', (SignatureGenerator.SIGNATURE_SIZE[0] * len(SignatureGenerator.LETTER_SET),
                                SignatureGenerator.SIGNATURE_SIZE[1]), 1)

        offset = 0

        for letter in SignatureGenerator.LETTER_SET:
            letter_image = Image.open('dump/letters/' + letter + '.png')

            image.paste(letter_image, (offset, 0))

            offset += SignatureGenerator.SIGNATURE_SIZE[0]

        return image

    @staticmethod
    def create_signatures(image):
        # Save the entire signature first
        image.save('dump/signatures/complete.png')

        # Now loop the image one signature level at the time
        for i in range(SignatureGenerator.SIGNATURE_SIZE[1]):
            SignatureGenerator.create_signature(image, i)

    @staticmethod
    def create_signature(image, signature_position=0, signature_height=1):
        signature_image = image.crop((0, signature_position, image.width, signature_position + signature_height))

        signature_image.save('dump/signatures/' + str(signature_position) + '.png')

if __name__ == '__main__':
    SignatureGenerator.generate()
