import tensorflow as tf
import os

with tf.device(tf.train.replica_device_setter(0, worker_device='/cpu:0')):
    filename_queue = tf.train.string_input_producer(tf.gfile.Glob('/nfs/data/dougals/lsun-32/*'), num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
#        'image/height': _int64_feature(im.height),
#        'image/width': _int64_feature(im.width),
#        'image/colorspace': _rgb,
#        'image/channels': _channels,
        'image/class/label': tf.FixedLenFeature([1], tf.int64),
#        'image/filename': _bytes_feature(name),
#        'image/db': _bytes_feature(db_name.encode('utf-8')),
#        'image/format': _jpeg,
        'image/encoded': tf.FixedLenFeature([], tf.string),
    })
    image = tf.image.decode_jpeg(features['image/encoded'])
   
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(image))

