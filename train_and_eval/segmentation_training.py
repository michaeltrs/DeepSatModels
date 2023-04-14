import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_model_data_input, get_loss_data_input
import argparse


def train_and_evaluate(net, dataloaders, config, device):

    def train_step(net, sample, loss_fn, optimizer, device, model_input_fn, loss_input_fn):
        optimizer.zero_grad()
        # model forward pass
        outputs = net(model_input_fn(sample, device))
        outputs = outputs.permute(0, 2, 3, 1)
        # model backward pass
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        # run optimizer
        optimizer.step()
        return outputs, ground_truth, loss
  
    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        Neval = len(evalloader)
        
        predicted_all = []
        labels_all = []
        losses_all = []
        
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                if step % 100 == 0:
                    print("Eval step %d of %d" % (step, Neval))

                logits = net(model_input_fn(sample, device))
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)  # .cpu().numpy()

                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                
                target, mask = ground_truth

                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())
        
        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "---------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "---------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = config['SOLVER']['lr_base']
    dr = config['SOLVER']['lr_decay']
    reset_lr = config['SOLVER']['reset_lr']
    reset_lr_at_epoch = get_params_values(config['SOLVER'], "reset_lr_at_epoch", False)
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint_path = config['CHECKPOINT']["load_from_checkpoint"]
    partial_restore = config['CHECKPOINT']["partial_restore"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    dr_epochs = get_params_values(config['SOLVER'], "dr_epochs", 2)
    restart_clock = get_params_values(config['CHECKPOINT'], "restart_clock", False)

    start_global = 1
    start_epoch = 1
    if checkpoint_path:
        checkpoint_path = load_from_checkpoint(net, checkpoint_path, partial_restore=partial_restore)
        if restart_clock:
            start_global = 1
            start_epoch = 1
        else:
            start_global = int(checkpoint_path.split("_")[-1].split(".")[0])
            start_epoch = int(checkpoint_path.split("_")[-2].split(".")[0])
        if not reset_lr:
            lr *= dr ** start_epoch
    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    model_input_fn = get_model_data_input(config)
    
    loss_input_fn = get_loss_data_input(config)
    
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}
    
    trainable_params = get_net_trainable_params(net)
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    
    decay_fn = lambda epoch: dr ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=decay_fn)

    writer = SummaryWriter(save_path)

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):

            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            
            logits, ground_truth, loss = train_step(net, sample, loss_fn, optimizer, device,
                                                    model_input_fn=model_input_fn, loss_input_fn=loss_input_fn)
            labels, unk_masks = ground_truth

            # save model ----------------------------------------------------------------------------------------------#
            if abs_step % save_steps == 0:
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%s_%d_%d.pth" % (save_path, save_path.split("/")[-1], epoch, abs_step))
                else:
                    torch.save(net.state_dict(), "%s/%s_%d_%d.pth" % (save_path, save_path.split("/")[-1], epoch, abs_step))

            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                      "batch recall: %.4f, batch F1: %.4f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                       batch_metrics['Recall'], batch_metrics['F1']))

            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval",
                                      optimizer=None)
                net.train()


        if epoch % dr_epochs == 0:
            scheduler.step()
            print("decaying lr, new lr: %.10f" % optimizer.param_groups[0]["lr"])

        if reset_lr_at_epoch and (epoch % reset_lr_at_epoch == 0):  # reset learn rate, useful for experiments in Germany
            scheduler = LambdaLR(optimizer, lr_lambda=decay_fn)

    print('Finished Training')
    

if __name__ == "__main__":
    
    #------------------------------------------------------------------------------------------------------------------#
    # USER INPUT
    parser = argparse.ArgumentParser(description='gather relative paths for MTLCC tfrecords')
    parser.add_argument('--config_file', type=str, default="configs/MTLCC/UNet3Df.yaml",
                        help='.yaml configuration file to use')
    parser.add_argument('--gpu_ids', type=list, default=["0", "1"], action='store',
                        help='gpu ids for running experiment')  # , required=True)
    opt = parser.parse_args()
    gpu_ids = opt.gpu_ids
    config_file = opt.config_file

    device_ids = [int(i) for i in gpu_ids if i.isnumeric()]
    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)
