from data.MTLCC.tfrecord2tif import TFRecord2Numpy
from utils.multiprocessing_utils import run_pool
import pickle
import numpy as np
import pandas as pd
import random
import os
from glob import glob
import argparse


def tfrec2pickle(paths, rootdir):

    abs_in_paths = [os.path.join(rootdir, p) for p in paths]
    tfrec2np = TFRecord2Numpy(abs_in_paths)

    files_saved = []
    for i, p in enumerate(paths):
        # p = paths[0]

        try:
            relative_path = "%s.pickle" % "/".join(p.split("/")[-2:]).split(".")[0]
            save_name = os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl", relative_path)

            if os.path.exists(save_name):
                files_saved.append(os.path.join("data_IJGI18/datasets/full/240pkl", relative_path))
                print("existing file %d of %d" % (i, len(paths)))
                continue

            print("processing file %d of %d" % (i, len(paths)))
            x10, x20, x60, day, year, labels = tfrec2np.tfrecord2npy()
            year = np.ones(day.shape[0]).astype(np.int32) * (2000 + int(p.split("/")[-2][4:]))  # All tfrecords have year 2017

            data = {"x10": x10.astype(np.int16),
                    "x20": x20.astype(np.int16),
                    "x60": x60.astype(np.int16),
                    "day": day.astype(np.int16),
                    "year": year.astype(np.int16),
                    "labels": labels.astype(np.int8)}

            if not os.path.isdir(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))

            with open(save_name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            files_saved.append(os.path.join("data_IJGI18/datasets/full/240pkl", relative_path))

        except:
            continue

    return np.array(files_saved)


def split_data_paths(pkl_paths, train_ids_file, eval_ids_file, rootdir):
    get_id = lambda s: s.split("/")[-1].split(".")[0]

    eval_ids = pd.read_csv(eval_ids_file, header=None)
    train_ids = pd.read_csv(train_ids_file, header=None)

    pkl_paths[1] = pkl_paths[0].apply(get_id)

    train_paths = pkl_paths[pkl_paths[1].isin(train_ids[0].astype(str))][0]
    eval_paths = pkl_paths[pkl_paths[1].isin(eval_ids[0].astype(str))][0]

    train_paths.to_csv(os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl/train_paths.csv"),
                       header=None, index=False)
    eval_paths.to_csv(os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl/eval_paths.csv"),
                      header=None, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='gather relative paths for MTLCC tfrecords')
    parser.add_argument('--rootdir', required=True, help='data root directory')
    parser.add_argument('--numworkers', type=int, default='4', help='number of parallel processes')
    opt = parser.parse_args()

    data = glob(os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240/data16/*.tfrecord.gz"))
    data = pd.DataFrame(["/".join(d.split("/")[-6:]) for d in data])
    data.to_csv(
        os.path.join(opt.rootdir, "data_IJGI18/datasets/full/tfrecords240_paths.csv"), header=None, index=False)

    paths = data[0].values.tolist()
    if not os.path.exists(os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240pkl")):
        os.makedirs(os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240pkl"))

    def extract_fun(paths):
        return tfrec2pickle(paths, opt.rootdir)  # , save_rootdir)

    out = run_pool(paths, extract_fun, opt.numworkers)

    pkl_paths = pd.DataFrame(np.concatenate(out))
    pkl_paths.to_csv(
        os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240pkl/data_paths.csv"), header=None, index=False)

    eval_ids_file = os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240/tileids/eval.tileids")
    test_ids_file = os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240/tileids/test_fold0.tileids")
    train_ids_file = os.path.join(opt.rootdir, "data_IJGI18/datasets/full/240/tileids/train_fold0.tileids")

    split_data_paths(pkl_paths, train_ids_file, eval_ids_file, opt.rootdir)
