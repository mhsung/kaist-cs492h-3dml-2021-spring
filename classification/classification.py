# Minhyuk Sung (mhsung@kaist.ac.kr)

from pointnet import PointNetCls

import argparse
import h5py
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--n_workers', type=int, default=4,
                    help='number of data loading workers')

parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
parser.add_argument('--step_size', type=int, default=20, help='step size')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

parser.add_argument('--in_data_file', type=str,
                    default='data/ModelNet/modelnet_classification.h5',
                    help="data directory")
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--out_dir', type=str, default='outputs',
                    help='output directory')
args = parser.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, point_clouds, class_ids):
        self.point_clouds = torch.from_numpy(point_clouds).float()
        self.class_ids = torch.from_numpy(class_ids).long()

    def __len__(self):
        return np.shape(self.point_clouds)[0]

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.class_ids[idx]


def create_datasets_and_dataloaders():
    assert(os.path.exists(args.in_data_file))
    f = h5py.File(args.in_data_file, 'r')

    train_data = Dataset(f['train_point_clouds'][:], f['train_class_ids'][:])
    test_data = Dataset(f['test_point_clouds'][:], f['test_class_ids'][:])

    n_classes = np.amax(f['train_class_ids']) + 1
    print('# classes: {:d}'.format(n_classes))

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=int(args.n_workers))

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=int(args.n_workers))

    return train_data, train_dataloader, test_data, test_dataloader, n_classes


def compute_loss(points, gt_classes, pred_class_logits):
    # points: (batch_size, n_points, dim_input)
    # gt_classes: (batch_size)
    # pred_class_logits: (batch_size, n_classes)
    loss = F.cross_entropy(input=pred_class_logits, target=gt_classes)
    return loss


def compute_accuracy(points, gt_classes, pred_class_logits):
    # points: (batch_size, n_points, dim_input)
    # gt_classes: (batch_size)
    # pred_class_logits: (batch_size, n_classes)
    pred_classes = pred_class_logits.max(1)[1]
    acc = float(pred_classes.eq(gt_classes).sum()) / gt_classes.size()[0] * 100
    return acc


def run_train(data, net, optimizer, writer=None):
    # Parse data.
    points, gt_classes = data
    points = points.cuda()
    gt_classes = gt_classes.cuda().squeeze()
    # points: (batch_size, n_points, dim_input)
    # gt_classes: (batch_size)

    # Reset gradients.
    # https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html#zero-the-gradients-while-training-the-network
    optimizer.zero_grad()

    # Predict.
    pred_class_logits = net.train()(points)

    # Compute the loss.
    loss = compute_loss(points, gt_classes, pred_class_logits)

    with torch.no_grad():
        # Compute the accuracy.
        acc = compute_accuracy(points, gt_classes, pred_class_logits)

    # Backprop.
    loss.backward()
    optimizer.step()

    return loss, acc


def run_eval(data, net, optimizer, writer=None):
    # Parse data.
    points, gt_classes = data
    points = points.cuda()
    gt_classes = gt_classes.cuda().squeeze()
    # points: (batch_size, n_points, dim_input)
    # gt_classes: (batch_size)

    with torch.no_grad():
        # Predict.
        pred_class_logits = net.eval()(points)

        # Compute the loss.
        loss = compute_loss(points, gt_classes, pred_class_logits)

        # Compute the accuracy.
        acc = compute_accuracy(points, gt_classes, pred_class_logits)

    return loss, acc


def run_epoch(dataset, dataloader, train, epoch=None, writer=None):
    total_loss = 0.0
    total_acc = 0.0
    n_data = len(dataset)

    # Create a progress bar.
    pbar = tqdm(total=n_data, leave=False)

    mode = 'Train' if train else 'Test'
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)

    for i, data in enumerate(dataloader):
        # Run one step.
        loss, acc = run_train(data, net, optimizer, writer) if train else \
                run_eval(data, net, optimizer, writer)

        if train and writer is not None:
            # Write results if training.
            assert(epoch is not None)
            step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/Train', loss, step)
            writer.add_scalar('Accuracy/Train', acc, step)

        batch_size = list(data[0].size())[0]
        total_loss += (loss * batch_size)
        total_acc += (acc * batch_size)

        pbar.set_description('{} {} Loss: {:f}, Acc : {:.2f}%'.format(
            epoch_str, mode, loss, acc))
        pbar.update(batch_size)

    pbar.close()
    mean_loss = total_loss / float(n_data)
    mean_acc = total_acc / float(n_data)
    return mean_loss, mean_acc


def run_epoch_train_and_test(
    train_dataset, train_dataloader, test_dataset, test_dataloader, epoch=None,
        writer=None):
    train_loss, train_acc = run_epoch(
        train_dataset, train_dataloader, train=args.train, epoch=epoch,
        writer=writer)
    test_loss, test_acc = run_epoch(
        test_dataset, test_dataloader, train=False, epoch=epoch, writer=None)

    if writer is not None:
        # Write test results.
        assert(epoch is not None)
        step = (epoch + 1) * len(train_dataloader)
        writer.add_scalar('Loss/Test', test_loss, step)
        writer.add_scalar('Accuracy/Test', test_acc, step)

    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)

    log = epoch_str + ' '
    log += 'Train Loss: {:f}, '.format(train_loss)
    log += 'Train Acc: {:.2f}%, '.format(train_acc)
    log += 'Test Loss: {:f}, '.format(test_loss)
    log += 'Test Acc: {:.2f}%.'.format(test_acc)
    print(log)


if __name__ == "__main__":
    print(args)

    # Load datasets.
    train_dataset, train_dataloader, test_dataset, test_dataloader, \
        n_classes = create_datasets_and_dataloaders()

    # Create the network.
    n_dims = 3
    net = PointNetCls(n_dims, n_classes)
    if torch.cuda.is_available():
        net.cuda()

    # Load a model if given.
    if args.model != '':
        net.load_state_dict(torch.load(args.model))

    # Set an optimizer and a scheduler.
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate,
        betas=(args.beta1, args.beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)

    # Create the output directory.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Train.
    if args.train:
        writer = SummaryWriter(args.out_dir)

        for epoch in range(args.n_epochs):
            run_epoch_train_and_test(
                train_dataset, train_dataloader, test_dataset, test_dataloader,
                epoch, writer)

            if (epoch + 1) % 10 == 0:
                # Save the model.
                model_file = os.path.join(
                    args.out_dir, 'model_{:d}.pth'.format(epoch + 1))
                torch.save(net.state_dict(), model_file)
                print("Saved '{}'.".format(model_file))

            scheduler.step()

        writer.close()
    else:
        run_epoch_train_and_test(
            train_dataset, train_dataloader, test_dataset, test_dataloader)
