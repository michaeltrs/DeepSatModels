from multiprocessing import Pool


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def run_pool(x, f, num_cores, split=False):
    if not split:
        x = split_num_segments(x, num_cores)
    # x = [[x_, i] for i, x_ in enumerate(x)]
    pool = Pool(num_cores)
    res = pool.map(f, x)
    return res


def split_num_segments(inlist, num_segments):
    chunk_size = len(inlist) // num_segments
    res = []
    for i in range(num_segments):
        if i == num_segments - 1:
            res.append(inlist[i * chunk_size:])
        else:
            res.append(inlist[i * chunk_size:(i + 1) * chunk_size])
    return res


def split_size_segments(inlist, seg_size):
    i = 0
    newlist = []
    while len(inlist) - len(newlist) * seg_size > seg_size:
        newlist.append(inlist[i * seg_size: (i + 1) * seg_size])
        i += 1
    if len(inlist) - len(newlist) * seg_size > 0:
        newlist.append(inlist[i * seg_size:])
    return newlist
