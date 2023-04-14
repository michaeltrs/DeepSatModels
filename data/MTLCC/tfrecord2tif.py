from data.MTLCC.S2parser import S2parser
import tensorflow as tf
import numpy as np


class TFRecord2Numpy:
    def __init__(self, paths):
        self.parser = S2parser()

        dataset = tf.data.TFRecordDataset(paths, compression_type="GZIP")
        self.dataset = dataset.map(self.mapping_function, num_parallel_calls=1)

        self.iterator = iter(self.dataset)

    def tfrecord2npy(self):
        x10, x20, x60, doy, year, labels = next(self.iterator)

        x10, x20, x60, doy, year, labels = [np.array(f) for f in [x10, x20, x60, doy, year, labels]]
        x10, x20, x60, doy, year, labels = remove_padded_instances(x10, x20, x60, doy, year, labels)
        return x10, x20, x60, doy, year, labels
    
    def mapping_function(self, serialized_feature):
        feature = self.parser.parse_example(serialized_example=serialized_feature)
        return feature


def remove_padded_instances(x10,x20,x60,doy,year,labels):

    # remove padded instances
    mask = doy > 0
    x10 = x10[mask]
    x20 = x20[mask]
    x60 = x60[mask]
    doy = doy[mask]
    year = year[mask]
    labels = labels[mask]

    return x10,x20,x60,doy,year,labels
