"""
This module was made for starting as subprocess. It us a file with arguments in the same fold
File name must be send in stdin 
"""

# import warnings
# warnings.filterwarnings('ignore')
import sys

if __name__ != '__main__':
	print('Error:This module was made for starting as subprocess', file=sys.stderr)
else:
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	import numpy as np
	import tensorflow as tf
	import tensorflow_io as tfio
	import pickle
	from multiprocessing.pool import Pool
	from joblib import Parallel, delayed

	# there must be the pkl-file name with dict of arguments in stdin
	f_args = open(sys.stdin.readline().rstrip(), 'rb')
	d_args, image_size, file_numb = pickle.load(f_args)
	f_args.close()
	tfr_file = f'data/tfrec/tfr{file_numb:05d}.tfrecords'

	

	def roi(image, image_size, percent = 95):
    
		image = tf.io.read_file(image)
		image = tfio.image.decode_dicom_image(image, scale='auto', on_error='lossy', dtype=tf.uint8)
		
		# ------------ commented out due to huge size of tfrec files 
		# np.std needs float but encoding_png needs uint8 or uint16
		image = tf.cast(tf.squeeze(image), tf.float32).numpy()
	    
		# standard deviation for excluding background
		row_mask = np.std(image, axis=1) > 0.5
		clmn_mask =  np.std(image, axis=0) > 0.5
		image = image[row_mask, :][:, clmn_mask]
		    
		# 95-th percentile, for excluding of on-image labels 
		row_mask = np.percentile(image, percent, axis=1) > 0.5
		clmn_mask =  np.percentile(image, percent, axis=0) > 0.5
		image = image[row_mask, :][:, clmn_mask]
		# ------------ commented out due to huge size of tfrec files 
		# image = tf.io.encode_jpeg(tf.squeeze(image, 0))
		image = tf.image.resize(tf.expand_dims(image, 2), image_size)
		image = tf.image.convert_image_dtype(image, tf.uint8)
		image = tf.io.encode_png(image)
		return image
	
	def _bytes_feature(value):
		"""Returns a bytes_list from a string / byte."""
		if isinstance(value, type(tf.constant(0))):
			value = tf.io.serialize_tensor(value).numpy() # BytesList won't unpack a string from an EagerTensor.

			# print('byte len - ', len(value))
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _float_feature(value):
		"""Returns a float_list from a float / double."""
		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

	def _int64_feature(value):
		"""Returns an int64_list from a bool / enum / int / uint."""
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def serialize_example(image, label):
		"""
		Creates a tf.train.Example message ready to be written to a file.
		"""
		# Create a dictionary mapping the feature name to the tf.train.Example-compatible
		# data type.
		feature = {
			'image': _bytes_feature(image),
			'label': _int64_feature(label),
			}
    
		# Create a Features message using tf.train.Example.
		example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

		return example_proto.SerializeToString()

	def make_rec(file, label):
    

		# 

		# for filename, label in data.values:
			#print(type(filename))
			#image = tfio.image.decode_dicom_image(image_bytes, scale='auto', on_error='lossy', dtype=tf.uint8)
		image = roi(file, image_size)
		# print('image.shape - ', image.shape)
		# print('image to tf.const type - ', type(tf.constant(image)))
		serialized_example = serialize_example(tf.constant(image), label)
		return serialized_example
		

	# for _, row in d_args.iterrows():
	# 	path2file = f'data/{row.patient_id}/{row.image_id}.dcm'
	# 	foo = roi(path2file)

	lst = [(f'data/{row.patient_id}/{row.image_id}.dcm', row.cancer) for _, row in d_args.iterrows()]
	print(lst[0])
	# poller = Pool()
	# poller.map(roi, lst)  #poor using of cpu...
	with tf.io.TFRecordWriter(tfr_file) as writer:
		
		for file, label in lst:
			se = make_rec(file, label)
		# print(type(serialized_example))
			writer.write(se)
		#-------------------------- can't write parallel :(
		# se = Parallel(n_jobs=os.cpu_count(), backend='threading')(delayed(make_rec)(file, label) for file, label in lst)
		# writer.write(se)


	#print('image shape is ', foo.shape)
	#print('image type is ', type(foo))

	print('child process id - ', os.getpid())
	sys.stdout.flush()
	os._exit(99)

