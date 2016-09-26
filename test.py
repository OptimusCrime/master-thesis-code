
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

# width, height
image_size = (50, 40)

constraints = {
	'left': {
		'value': None,
		'letter': None
	},
	'right': {
		'value': None,
		'letter': None
	},
	'top': {
		'value': None,
		'letter': None
	},
	'bottom': {
		'value': None,
		'letter': None
	}
}

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for letter in letters:
	img = Image.new('1', image_size, 1)
	draw = ImageDraw.Draw(img)

	font = ImageFont.truetype("arial-mono.ttf", 35)
	draw.text((0, 0), letter, font=font, fill=0)

	arr = np.fromiter(list(img.getdata()), dtype="int").reshape((image_size[1], image_size[0]))

	for i in range(image_size[1]):
		for j in range(image_size[0]):
			if arr[i][j] == 0:
				# left
				if constraints['left']['value'] is None or constraints['left']['value'] > j:
					constraints['left']['value'] = j
					constraints['left']['letter'] = letter

				# right
				if constraints['right']['value'] is None or constraints['right']['value'] < j:
					constraints['right']['value'] = j
					constraints['right']['letter'] = letter

				# top
				if constraints['top']['value'] is None or constraints['top']['value'] > i:
					constraints['top']['value'] = i
					constraints['top']['letter'] = letter

				# bottom
				if constraints['bottom']['value'] is None or constraints['bottom']['value'] < i:
					constraints['bottom']['value'] = i
					constraints['bottom']['letter'] = letter

	del img
	#print(list(arr))

	#for line in arr:
	#	print (list(line))
	#print('')
	#img.save('sample-out.png')

#print(constraints)

crop_size = tuple([constraints['left']['value'], constraints['top']['value'], constraints['right']['value'], constraints['bottom']['value']])	
print(crop_size)
for letter in letters:
	img = Image.new('1', image_size, 1)
	draw = ImageDraw.Draw(img)

	font = ImageFont.truetype("arial-mono.ttf", 35)
	draw.text((0, 0), letter, font=font, fill=0)

	img = img.crop(crop_size)
	img.save('letters/' + letter + '.png')

img = Image.new('1', image_size, 1)
draw = ImageDraw.Draw(img)

font = ImageFont.truetype("arial-mono.ttf", 35)
for letter in letters:
	draw.text((0, 0), letter, font=font, fill=0)

img = img.crop(crop_size)
img.save('letters/all.png')
