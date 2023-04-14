def write_mean_summaries(writer, metrics, abs_step, mode="train", optimizer=None):
    for key in metrics:
        writer.add_scalars(main_tag=key, tag_scalar_dict={'%s_Average' % mode: metrics[key]},
                           global_step=abs_step, walltime=None)
    if optimizer is not None:
        writer.add_scalar('learn_rate', optimizer.param_groups[0]["lr"], abs_step)


def write_class_summaries(writer, metrics, abs_step, mode="eval", optimizer=None):
    unique_labels, metrics = metrics
    print("saving per class summaries")
    for key in metrics:
        tag_scalar_dict = {'%s_%s' % (mode, str(i)): val for i, val in zip(unique_labels, metrics[key])}
        writer.add_scalars(main_tag=key, tag_scalar_dict=tag_scalar_dict, global_step=abs_step, walltime=None)
    if optimizer is not None:
        writer.add_scalar('learn_rate', optimizer.param_groups[0]["lr"], abs_step)


def write_histogram_summaries(writer, metrics, abs_step, mode="train"):
    for key in metrics:
        writer.add_histogram("%s_%s" % (mode, key), metrics[key], global_step=abs_step)
