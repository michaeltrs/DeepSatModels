from data.MTLCC.S2parser import S2parser
import tensorflow as tf
import numpy as np
import rasterio
import pandas as pd
import os
import datetime

# tf.compat.v1.disable_eager_execution()

class TFRecord2Numpy:
    def __init__(self, paths):
        # self.paths = paths
        self.parser = S2parser()
        # self.sess = tf.compat.v1.InteractiveSession()

        # self.sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer(),
        #               tf.compat.v1.tables_initializer()])
        dataset = tf.data.TFRecordDataset(paths, compression_type="GZIP")
        self.dataset = dataset.map(self.mapping_function, num_parallel_calls=1)
        # sess = tf.InteractiveSession()
        # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

        # self.iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)  # .make_initializable_iterator()
        # self.sess.run([self.iterator.initializer])
        self.iterator = iter(self.dataset)  # .make_initializable_iterator()

    def tfrecord2npy(self):
        # sess.run([iterator.initializer])
        # x10, x20, x60, doy, year, labels = self.sess.run(self.iterator.get_next())
        x10, x20, x60, doy, year, labels = next(self.iterator)

        x10, x20, x60, doy, year, labels = [np.array(f) for f in [x10, x20, x60, doy, year, labels]]
        x10, x20, x60, doy, year, labels = remove_padded_instances(x10, x20, x60, doy, year, labels)
        return x10, x20, x60, doy, year, labels
    
    def mapping_function(self, serialized_feature):
        # read data from .tfrecords
        feature = self.parser.parse_example(serialized_example=serialized_feature)
        return feature
    


# def tfrecord2npy(path):
#
#     parser = S2parser()
#     sess = tf.InteractiveSession()
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
#
#     def mapping_function(serialized_feature):
#         # read data from .tfrecords
#         feature = parser.parse_example(serialized_example=serialized_feature)
#         return feature
#
#     dataset = tf.data.TFRecordDataset([path], compression_type="GZIP")
#
#     dataset = dataset.map(mapping_function, num_parallel_calls=1)
#     # sess = tf.InteractiveSession()
#     # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
#     iterator = dataset.make_initializable_iterator()
#     sess.run([iterator.initializer])
#     x10, x20, x60, doy, year, labels = sess.run(iterator.get_next())
#
#     x10, x20, x60, doy, year, labels = [np.array(f) for f in [x10, x20, x60, doy, year, labels]]
#
#     x10, x20, x60, doy, year, labels = remove_padded_instances(x10, x20, x60, doy, year, labels)
#
#     return x10, x20, x60, doy, year, labels


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


# def write_tif(arr, filename, geo=None):
#     """
#     Write geotiff file
#     :param arr: image array, size (H, W, Nchannels)
#     :param filename: save filename for geotiff
#     :param geo: geolocation info, ["north", "west", "pixelx", "pixely", "crs"]
#                 crs: coordinate reference system
#                 example: [635000.0, 5345480.0, 10, -10, 32632]
#     :return:
#     """
#     path = os.path.dirname(filename)
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     print("writing "+filename)
#
#     # geo = gt.north, gt.west, pixelx, pixely, crs
#     if geo is not None:
#         north, west, pixelx, pixely, crs = geo
#         crs = crs
#         transform = rasterio.transform.from_origin(north, west, pixelx, pixely)
#     else:
#         transform = None
#         crs = None
#
#     H, W, D = arr.shape
#
#     new_dataset = rasterio.open(
#         filename,
#         'w',
#         driver='GTiff',
#         height=H,
#         width=W,
#         count=D,
#         dtype=rasterio.uint16,
#         crs=crs,
#         transform=transform,
#     )
#     for d in range(D):
#         new_dataset.write(arr[:, :, d].astype(np.uint16), d+1)
#
#     new_dataset.close()
#
#
# def write_timeseries(tfrecordgzpath, outdir, geotransforms=None):
#
#     x10, x20, x60, doy, year, labels = tfrecord2npy(tfrecordgzpath)
#
#     id = int(os.path.basename(tfrecordgzpath).replace(".tfrecord.gz", ""))
#
#     if geotransforms is not None:
#         print("reading geotranform file from " + geotransforms)
#         df = pd.read_csv(geotransforms, index_col=0,
#                          names=["north", "pixelx", "shearx", "west", "sheary", "pixely", "crs"])
#         g = df.loc[id]
#
#         north = g.north
#         west = g.west
#         crs = int(g.crs)
#     else:
#         north = 0
#         west = 0
#         crs = None
#
#     outpath = os.path.join(outdir, str(id))
#
#     if not os.path.exists(outpath):
#         os.makedirs(outpath)
#
#     for t in range(x10.shape[0]):
#         y = year[t]
#         d = doy[t]
#         dt = datetime.datetime.strptime(str(y) +' '+ str(d) , '%Y %j')
#
#         date = dt.strftime("%Y%m%d")
#
#         write_tif(x10[t], os.path.join(outpath, "10m", date + ".tif"), geo=(north, west, 10, 10, crs))
#         write_tif(x20[t], os.path.join(outpath, "20m", date + ".tif"), geo=(north, west, 20, 20, crs))
#         write_tif(x60[t], os.path.join(outpath, "60m", date + ".tif"), geo=(north, west, 60, 60, crs))
#
#     write_tif(labels[0][:, :, None], os.path.join(outpath, "labels.tif"), geo=(north, west, 10, 10, crs))
#
#
# def main():
#     #parser = argparse.ArgumentParser(description='convert tfrecord.gz file to folder of tif images.')
#     #parser.add_argument('tfrecord', help='path to tfrecord.gz file (format path/0000.tfrecord.gz')
#     #parser.add_argument('--outdir', default="tif", help='output directory')
#     #parser.add_argument('--geotransforms', default=None, help='path to csv file for geotransforms.')
#     #args = parser.parse_args()
#
#     tfrecordgzpath = '/home/michaeltrs/Data/SatelliteImagery/MTLCC_prev/data_IJGI18/datasets/full/480/data16/1.tfrecord.gz'
#     outdir = '/home/michaeltrs/Data/SatelliteImagery/MTLCC_prev/tests_folder'
#     geotransforms = '/home/michaeltrs/Data/SatelliteImagery/MTLCC_prev/data_IJGI18/datasets/full/480/geotransforms.csv'
#     #tfrecordgzpath = "/data/data_IJGI18/datasets/full/480/data16/5886.tfrecord.gz"
#     #outdir = "0"
#     #geotransforms = "/data/data_IJGI18/datasets/full/480/geotransforms.csv"
#     #geotransforms = None
#
#     write_timeseries(tfrecordgzpath, outdir, geotransforms)
#
#
# if __name__ == "__main__":
#     main()
#     # pass

#-----------------------------------------------------------------------------------------------------------#
"""
from PIL import Image
im = Image.open('/home/michaeltrs/Data/SatelliteImagery/MTLCC_prev/tests_folder/1/10m/20160103.tif')
im.show()


import matplotlib.pyplot as plt
I = plt.imread('/home/michaeltrs/Data/SatelliteImagery/MTLCC_prev/tests_folder/1/10m/20160103.tif')


import georasters as gr

# Load data
raster = './data/slope.tif'
data = gr.from_file(raster)


import rasterio
import rasterio.features
import rasterio.warp

geotif_path = '/home/michaeltrs/Data/SatelliteImagery/MTLCC_prev/tests_folder/1/10m/20160103.tif'
with rasterio.open(geotif_path) as dataset:
    
    dataset = rasterio.open(geotif_path)
    
    dataset.name
    dataset.mode
    dataset.closed
    dataset.indexes
    dataset.dtypes
    dataset.nodatavals
    
    
    i = 1
    
    array = dataset.read(i)
    print(i, array.shape)

    plt.imshow(array)
    
    i += 1
    
    
    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()

    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):

        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            dataset.crs, 'EPSG:4326', geom, precision=6)

        # Print GeoJSON shapes to stdout.
        print(geom)
"""