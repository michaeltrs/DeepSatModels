import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from utils.torch_utils import load_from_checkpoint
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device
from data import get_dataloaders
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_histogram_summaries
from data import get_model_data_input, get_loss_data_input
import argparse


def train_and_evaluate(net, dataloaders, config, device):
    
    def train_step(net, sample, loss_fn, optimizer, device, model_input_fn, loss_input_fn):
        # zero the parameter gradients
        optimizer.zero_grad()
        # model forward pass
        outputs = net(model_input_fn(sample, device))
        # model backward pass
        ground_truth = loss_input_fn(sample, device)#[1]
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        # run optimizer
        optimizer.step()
        return outputs, ground_truth, loss

    def evaluate(net, evalloader, loss_fn):  # , config):
        Neval = len(evalloader)
        
        logits_bin_all = []
        samelabels_bin_all = []
        losses_all = []

        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                if step % 100 == 0:
                    print("Eval step %d of %d" % (step, Neval))
        
                logits = net(model_input_fn(sample, device))

                ground_truth = loss_input_fn(sample, device)
                if type(ground_truth) in [list, tuple]:
                    mask = ground_truth[1].clone()
                    sameclass_labels = ground_truth[0][mask]
                    logits = logits[mask]
                else:
                    sameclass_labels = ground_truth

                loss = loss_fn['all'](logits, sameclass_labels)
                
                logits_bin_all.append(logits.reshape(-1).cpu())
                samelabels_bin_all.append(sameclass_labels.reshape(-1).cpu())
                losses_all.append(loss.reshape(-1).cpu())

        print("-------------------------------------------------------------------------------------------------------")
        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        logits = torch.cat(logits_bin_all)
        ground_truth = torch.cat(samelabels_bin_all)
        losses = torch.cat(losses_all)
        mean_loss = losses.mean()

        logits0 = logits[ground_truth == 0].detach().cpu()
        logits1 = logits[ground_truth == 1].detach().cpu()
        print("EVAL, abs_step: %d, epoch: %d, step: %d, mean target %.6f, loss: %.4f, mean 0/1: %.4f/%.4f" %
              (abs_step, epoch, step, ground_truth.mean(), mean_loss, float(logits0.mean()), float(logits1.mean())))
        print("-------------------------------------------------------------------------------------------------------")

        return mean_loss, logits0, logits1

    #------------------------------------------------------------------------------------------------------------------#
    num_epochs = config['SOLVER']['num_epochs']
    lr = config['SOLVER']['lr_base']
    dr = config['SOLVER']['lr_decay']
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint_path = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    dr_epochs = get_params_values(config['SOLVER'], "dr_epochs", 1)

    if checkpoint_path:
        load_from_checkpoint(net, checkpoint_path, True)
        start_global = int(checkpoint_path.split("_")[-1].split(".")[0])
        start_epoch = int(checkpoint_path.split("_")[-2].split(".")[0])
        lr *= dr ** start_epoch
        print("current learn rate: ", lr)
    else:
        start_global = 1
        start_epoch = 1
    
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)
    
    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    model_input_fn = get_model_data_input(config)
    
    loss_input_fn = get_loss_data_input(config)
    
    loss_fn = {'all': get_loss(config, device, reduction='none'),
               'mean': get_loss(config, device, reduction="mean")}
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
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
            
            # save model ----------------------------------------------------------------------------------------------#
            if abs_step % save_steps == 0:  # evaluate model every eval_steps batches
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%s_%d_%d.pth" % (save_path, save_path.split("/")[-1], epoch, abs_step))
                else:
                    torch.save(net.state_dict(), "%s/%s_%d_%d.pth" % (save_path, save_path.split("/")[-1], epoch, abs_step))

            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:
                if type(ground_truth) in [list, tuple]:
                    labels, masks = ground_truth
                    logits = logits[masks]
                    ground_truth = 1 - labels[masks]
                logits0 = logits[ground_truth == 0].detach().cpu()
                logits1 = logits[ground_truth == 1].detach().cpu()
                print("abs_step: %d, epoch: %d, step: %d, mean target %.6f, loss: %.4f, mean 0/1: %.4f/%.4f" %
                    (abs_step, epoch, step, ground_truth.mean(), loss, float(logits0.mean()), float(logits1.mean())))
                write_mean_summaries(writer,
                                     {'loss': loss, 'mean_sim_0': logits0.mean(), 'mean_sim_1': logits1.mean()},
                                     abs_step, mode="train", optimizer=optimizer)
                write_histogram_summaries(writer, {0: logits0, 1: logits1}, abs_step, mode="train")

            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                mean_loss, logits0, logits1 = evaluate(net, dataloaders['eval'], loss_fn)  # , config)
                write_mean_summaries(writer,
                                     {"loss": mean_loss, 'mean_sim_0': logits0.mean(), 'mean_sim_1': logits1.mean()},
                                     abs_step, mode="eval", optimizer=None)
                write_histogram_summaries(writer, {0: logits0, 1: logits1}, abs_step, mode="eval")
                print("saved")
                net.train()

        if epoch % dr_epochs == 0:
            scheduler.step()
            print("decaying lr, new lr: %.10f" % optimizer.param_groups[0]["lr"])
        
    print('Finished Training')
    
    
if __name__ == "__main__":
    
    #------------------------------------------------------------------------------------------------------------------#
    # USER INPUT
    parser = argparse.ArgumentParser(description='PyTorch CSCL pre-training')
    parser.add_argument('--config_file', type=str, default="configs/MTLCC/UNet3Df_CSCL.yaml",
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
