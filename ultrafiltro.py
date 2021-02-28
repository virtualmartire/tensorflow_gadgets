###### ULTRAFILTRO: A TensorFlow's gadget for computer vision's datasets' construction by Stefano Martire, www.stefanomartire.it ######

"""
The buttafuoris are the dudes outside italian clubs with job to keep bad guys out. The buttafuori() function below transforms all the files in main_dir (already organised
in subdir corresponding to classes) in the jpeg format, giving them comfy names too ("classname_counter").

The ultrafiltro() function takes as input the aforesaid main_dir with all its files having exactly the jpeg extension and return a tensorflow dataset labelled with the
names corresponding to each class (i.e. to each directory name). Labels are in one-hot format.

The configure_for_training() function prepares our dataset for training with canonical preprocessing techniques.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pathlib
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def get_label(file_path):
	#convert the path to a list of path components
	parts = tf.strings.split(file_path, os.path.sep)
	#the second to last is the class-directory; it is returned one-hotted
	return parts[-2] == class_names

def decode_img(img, transform):
	#convert the compressed string to a 3D uint8 tensor
	img = tf.image.decode_jpeg(img, channels=3)
	#use 'convert_image_dtype' for convertion to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
	#apply the desired transformation and then return the output
	return transform(img)

def process_path(file_path, transform):
	label = get_label(file_path)
	#load the raw data from the file as a string
	img = tf.io.read_file(file_path)
	img = decode_img(img, transform)
	return img, label

#
##
###
##
#

def buttafuori(main_dir):
	data_dir = pathlib.Path(main_dir)

	categories = [item.name for item in data_dir.glob("*") if item.name != ".DS_Store"]
	print("Categories found:\n{}".format(categories))

	for categ in categories:
		array = [
					elem.name for elem in data_dir.glob( "{}/*.*".format(categ) )
				]

		k = 1
		for elem in array:
			new_name = "{}_{}.jpeg".format(categ, k)
			image = Image.open( main_dir+"/{}/{}".format(categ, elem) )
			if image.mode in ('RGBA', 'LA'):
				background = Image.new(image.mode[:-1], image.size)
				background.paste(image, image.split()[-1])
				image = background
			image.save( main_dir+"/{}/{}".format(categ, new_name) )
			os.remove(main_dir+"/{}/{}".format(categ, elem))
			k += 1
		print("{} done.".format(categ))

def ultrafiltro(main_dir, transform, show = True):
	#transform is the function that we want to apply to each of our sample (primarily to make them all geometrically equal)
	#when show == True a set of 9 transformed random samples are shown

	data_dir = pathlib.Path(main_dir)

	image_count = len(list(data_dir.glob("*/*.jpeg")))
	print( "\nultrafiltro found {} samples in the dataset.".format(image_count) )

	global class_names
	class_names = np.array(
					[item.name for item in data_dir.glob("*") if item.name != ".DS_Store"]
						)
	class_samples = [
					len(list(data_dir.glob( "{}/*.jpeg".format(class_) )))
						for class_ in class_names
					]
	class_dict = {
					class_:samples for class_, samples in zip(class_names, class_samples)
					}
	print( "\nThese are the {} categories detected:".format(len(class_names)) )
	print(class_dict)
	print()

	#dataset of the file paths
	list_ds = tf.data.Dataset.list_files(str(data_dir/"*/*.jpeg"))
	#final dataset with all the labels. Set 'num_parallel_calls' so multiple images are loaded/processed in parallel
	AUTOTUNE = tf.data.AUTOTUNE
	labeled_ds = list_ds.map(lambda x: process_path(x, transform), num_parallel_calls=AUTOTUNE)

	#the first 9 samples
	if show == True:
		show_ds = iter(labeled_ds)
		plt.figure(figsize=(10, 10))
		for i in range(9):
			x, y = next(show_ds)
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(x.numpy().astype("float32"))
			plt.title(y.numpy())
			plt.axis("off")
		plt.show()

	return labeled_ds

def configure_for_training(ds, train_prop, batch_size, buffer_size):

	image_count = int(tf.data.experimental.cardinality(ds))

	full_dataset = ds.shuffle(buffer_size=buffer_size)

	train_size = int(train_prop * image_count)
	val_size = int( (1 - train_prop) * image_count )

	train_ds = full_dataset.take(train_size)
	val_ds = full_dataset.skip(train_size)
	val_ds = val_ds.take(val_size)

	train_ds = train_ds.batch(batch_size, drop_remainder=True)
	val_ds = val_ds.batch(batch_size)

	return train_ds, val_ds