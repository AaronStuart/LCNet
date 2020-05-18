import argparse
import os

import torch
import torchvision
from torch import optim

from dataset.apollo import ApolloDaliDataset, ApolloBalanceTrainDataLoader
from loss.LossFactory import LossFactory
from scripts.visualize import Visualize

parser = argparse.ArgumentParser()
#############  Model  #############
parser.add_argument("--num_classes", type=int, default=38)

#############  Data  #############
parser.add_argument("--dataset_root_dir", type=str, default="/media/stuart/data/dataset/Apollo/Lane_Detection")
# parser.add_argument("--train_file", type=str, default='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray.txt')
parser.add_argument("--train_file", type=str, default='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray_split_by_class.json')
parser.add_argument("--num_threads", type=int, default=8)

#############  Loss  #############
parser.add_argument("--use_metric_loss", type=bool, default=False)
parser.add_argument("--metric_loss_weight", type=float, default=1)

parser.add_argument("--use_cluster_loss", type=bool, default=True)
parser.add_argument("--cluster_loss_weight", type=float, default=1)

############# Train  #############
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--warm_up_iters", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--pretrained_weights", type=str, default='/media/stuart/data/weights/DeepLabV3/cluster_foreground_with_ignored_balance_train_iter_100000_pretrained.pth')
parser.add_argument("--save_interval", type=int, default=1000, help="How many iterations are saved once?")
parser.add_argument("--visualize_interval", type=int, default=10, help="How many iterations are visualized once?")
parser.add_argument("--log_dir", type=str, default='/media/stuart/data/events')
parser.add_argument("--weights_save_dir", type=str, default='/media/stuart/data/weights')
parser.add_argument("--use_dali", type=bool, default=False)
args = parser.parse_args()
print(args)


class Train(object):
    def __init__(self, num_class,
                 use_metric_loss, metric_loss_weight,
                 use_cluster_loss, cluster_loss_weight,
                 weights_save_dir, use_dali
        ):
        """"""
        self.num_class = num_class

        self.use_metric_loss = use_metric_loss
        self.metric_loss_weight = metric_loss_weight

        self.use_cluster_loss = use_cluster_loss
        self.cluster_loss_weight = cluster_loss_weight

        self.weights_save_dir = weights_save_dir

        self.use_dali = use_dali

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.get_model('DeepLabV3', args.num_classes).to(self.device).train()

        self.optimizer = self.get_optimizer(args.learning_rate)

        self.loss = self.get_loss()

        self.dataloader = self.get_dataloader(
            dataset_dir=args.dataset_root_dir,
            file_path=args.train_file,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            is_train=True, use_dali=use_dali
        )

        self.visualizer = self.get_visualizer(args.log_dir)

    def get_model(self, model_name, num_classes):
        """"""
        assert model_name in ['DeepLabV3', 'FCN'], "model name doesn't exist"

        if model_name == 'DeepLabV3':
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                pretrained=False,
                num_classes=num_classes
            )
        if model_name == 'FCN':
            model = torchvision.models.segmentation.fcn_resnet50(
                pretrained=False,
                num_classes=num_classes
            )

        return model

    def get_optimizer(self, learning_rate=0.01):
        """"""
        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=learning_rate
        )

        return optimizer

    def get_loss(self):
        """"""
        loss = LossFactory(
            num_classes=self.num_class,
            use_metric_loss=self.use_metric_loss,
            metric_loss_weight=self.metric_loss_weight,
            use_cluster_loss=self.use_cluster_loss,
            cluster_loss_weight=self.cluster_loss_weight
        )

        return loss

    def get_dataloader(self, use_dali, dataset_dir, file_path, batch_size=1, num_threads=1, is_train=True):
        """"""
        if not os.path.exists(dataset_dir):
            print("%s doesn't exist." % dataset_dir)
            return
        if not os.path.exists(file_path):
            print("%s doesn't exist." % file_path)
            return

        if use_dali:
            # Get Dali dataloader
            train_dataloader = ApolloDaliDataset(
                root_dir=dataset_dir,
                file_path=file_path,
                batch_size=batch_size,
                num_threads=num_threads,
                is_train=is_train
            ).getIterator()
        else:
            train_dataloader = ApolloBalanceTrainDataLoader(
                root_dir=dataset_dir,
                json_path=file_path,
                batch_size = batch_size
            )

        return train_dataloader

    def get_visualizer(self, log_dir):
        """"""
        visualizer = Visualize(
            log_dir=os.path.join(log_dir, self.model.__class__.__name__),
            model=self.model
        )

        return visualizer

    def load_checkpoint(self, ckpt_path):
        """"""
        if not os.path.exists(ckpt_path):
            print("%s not exist, train from scratch." % ckpt_path)
            return

        ckpt_path = os.path.abspath(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']

        torch.save(self.model, '/home/stuart/PycharmProjects/LCNet/weights/DeepLabV3/model_cluster_foreground_with_ignored_balance_train_iter_100000_pretrained.pth')
        print("Load from %s successfully." % ckpt_path)
        return iteration

    def save_checkpoint(self, iter):
        """"""
        ckpt_dir = os.path.join(os.path.abspath(self.weights_save_dir), self.model.__class__.__name__)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # save model weights, optimizer status, and train iteration
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iter
        }
        ckpt_path = os.path.join(ckpt_dir, 'cluster_foreground_with_ignored_balance_train_iter_%d_pretrained.pth' % iter)

        torch.save(save_dict, ckpt_path)
        print("Save checkpoint to %s" % ckpt_path)

    def run(self):
        """"""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        iter = 0

        # load coco pretrained weights
        self.model.load_state_dict(
            torch.load('/home/stuart/.cache/torch/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'),
            strict = False
        )

        if args.pretrained_weights:
            iter = self.load_checkpoint(args.pretrained_weights)

        while True:
            for data in self.dataloader:
                iter += 1
                if self.use_dali:
                    # Note: DALI's RGB image have changed to BGR sequence, after go through Pytorch, maybe a bug
                    input, label = data[0]['input'][:, [2, 1, 0], :, :].to(self.device), data[0]['label'].to(self.device)
                else:
                    input, label = data[0], data[1]
                    input = torch.tensor(input, dtype = torch.float).to(self.device)
                    label = torch.tensor(label, dtype = torch.uint8).to(self.device)

                # run forward
                self.optimizer.zero_grad()
                logits = self.model(input)['out']

                # compute loss, backward, update weights
                loss = self.loss.compute_loss(logits, label, iter, args.warm_up_iters)
                loss['total_loss'].backward()
                self.optimizer.step()

                learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('iter %d: lr = %.5f, ' % (iter, learning_rate), loss)

                # visualize train process
                if iter % args.visualize_interval == 0:
                    self.visualizer.train_update(
                        iteration=iter,
                        learning_rate=learning_rate,
                        input=input[0].cpu(),
                        label=label[0].cpu(),
                        logits=logits[0].detach().cpu(),
                        loss=loss
                    )

                # save checkpoint
                if iter % args.save_interval == 0:
                    self.save_checkpoint(iter)


if __name__ == '__main__':
    Train(
        num_class=args.num_classes,
        use_metric_loss=args.use_metric_loss,
        metric_loss_weight=args.metric_loss_weight,
        use_cluster_loss=args.use_cluster_loss,
        cluster_loss_weight=args.cluster_loss_weight,
        weights_save_dir=args.weights_save_dir,
        use_dali = args.use_dali
    ).run()
