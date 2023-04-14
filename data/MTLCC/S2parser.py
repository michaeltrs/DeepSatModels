import tensorflow as tf
import numpy as np
import sys
import os

class S2parser():
    """ defined the Sentinel 2 .tfrecord format """
    def __init__(self):

        self.feature_format= {
            'x10/data': tf.io.FixedLenFeature([], tf.string),
            'x10/shape': tf.io.FixedLenFeature([4], tf.int64),
            'x20/data': tf.io.FixedLenFeature([], tf.string),
            'x20/shape': tf.io.FixedLenFeature([4], tf.int64),
            'x60/data': tf.io.FixedLenFeature([], tf.string),
            'x60/shape': tf.io.FixedLenFeature([4], tf.int64),
            'dates/doy': tf.io.FixedLenFeature([], tf.string),
            'dates/year': tf.io.FixedLenFeature([], tf.string),
            'dates/shape': tf.io.FixedLenFeature([1], tf.int64),
            'labels/data': tf.io.FixedLenFeature([], tf.string),
            'labels/shape': tf.io.FixedLenFeature([3], tf.int64)
        }

        return None

    def write(self, filename, x10, x20, x60, doy, year, labels):
        # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

        writer = tf.python_io.TFRecordWriter(filename)

        x10=x10.astype(np.int64)
        x20=x20.astype(np.int64)
        x60=x60.astype(np.int64)
        doy=doy.astype(np.int64)
        year=year.astype(np.int64)
        labels=labels.astype(np.int64)

        # Create a write feature
        feature={
            'x10/data' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[x10.tobytes()])),
            'x10/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x10.shape)),
            'x20/data' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[x20.tobytes()])),
            'x20/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x20.shape)),
            'x60/data' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[x60.tobytes()])),
            'x60/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x60.shape)),
            'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
            'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
            'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
            'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
            'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
        }


        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def get_shapes(self, sample):
        print("reading shape of data using the sample "+sample)
        data = self.read_and_return(sample)
        return [tensor.shape for tensor in data]

    def parse_example(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train_and_eval.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """

        feature = tf.io.parse_single_example(serialized_example, self.feature_format)
        # decode and reshape x10
        x10 = tf.reshape(tf.io.decode_raw(feature['x10/data'], tf.int64),tf.cast(feature['x10/shape'], tf.int32))

        x20 = tf.reshape(tf.io.decode_raw(feature['x20/data'], tf.int64), tf.cast(feature['x20/shape'], tf.int32))
        x60 = tf.reshape(tf.io.decode_raw(feature['x60/data'], tf.int64), tf.cast(feature['x60/shape'], tf.int32))

        doy = tf.reshape(tf.io.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
        year = tf.reshape(tf.io.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

        labels = tf.reshape(tf.io.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

        return x10, x20, x60, doy, year, labels

    def read(self, filenames):
        """ depricated! """

        if isinstance(filenames,list):
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        elif isinstance(filenames,tf.FIFOQueue):
            filename_queue = filenames
        else:
            print("please insert either list or tf.FIFOQueue")

        reader = tf.TFRecordReader()
        f, serialized_example = reader.read(filename_queue)

        print(f)

        feature = tf.parse_single_example(serialized_example, features=self.feature_format)

        # decode and reshape x10
        x10 = tf.reshape(tf.decode_raw(feature['x10/data'], tf.int64),tf.cast(feature['x10/shape'], tf.int32))

        x20 = tf.reshape(tf.decode_raw(feature['x20/data'], tf.int64), tf.cast(feature['x20/shape'], tf.int32))
        x60 = tf.reshape(tf.decode_raw(feature['x60/data'], tf.int64), tf.cast(feature['x60/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

        return x10, x20, x60, doy, year, labels

    def tfrecord_to_pickle(self,tfrecordname,picklename):
        try:
            import cPickle as pickle
        except:
            import pickle

        reader = tf.TFRecordReader()

        # read serialized representation of *.tfrecord
        filename_queue = tf.train.string_input_producer([tfrecordname], num_epochs=None)
        filename_op, serialized_example = reader.read(filename_queue)
        feature = self.parse_example(serialized_example)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            feature = sess.run(feature)

            coord.request_stop()
            coord.join(threads)

        pickle.dump(feature, open(picklename, "wb"), protocol=2)

    def read_and_return(self,filename):
        """ depricated! """

        # get feature operation containing
        feature_op = self.read([filename])

        with tf.Session() as sess:

            tf.global_variables_initializer()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            return sess.run(feature_op)

if __name__=='__main__':

    parser = S2parser()

    parser.tfrecord_to_pickle("1.tfrecord", "1.pkl")
