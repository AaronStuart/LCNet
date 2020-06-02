import argparse
import os

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader

from dataset.apollo import ApolloBalanceTrainDataLoader, ApolloDataset
from loss.LossFactory import LossFactory
from scripts.visualize import Visualize

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
#############  Model  #############
parser.add_argument("--model_name", type=str, default='DeepLabV3')
parser.add_argument("--num_classes", type=int, default=38)

#############  Dataset  #############
parser.add_argument("--dataset_root_dir", type=str, default="/media/stuart/data/dataset/Apollo/Lane_Detection")
parser.add_argument("--train_file", type=str, default='dataset/apollo_train_gray.txt')
parser.add_argument("--val_file", type=str, default='dataset/apollo_val_gray.txt')
parser.add_argument("--balance_train_file", type=str, default='/home/stuart/PycharmProjects/LCNet/dataset/apollo_train_gray_balance_train.json')

#############  DataLoader  #############
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_threads", type=int, default=12)

#############  Loss  #############
parser.add_argument("--use_cluster_loss", type=bool, default=True)
parser.add_argument("--cluster_loss_weight", type=float, default=1)

############# Train  #############
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--warm_up_iters", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--resume", type=str, default='/media/stuart/data/weights/DeepLabV3/balance_train_cluster_loss_iter_82000.pth')
parser.add_argument("--save_interval", type=int, default=1000, help="How many iterations are saved once?")
parser.add_argument("--visualize_interval", type=int, default=100, help="How many iterations are visualized once?")
parser.add_argument("--output_dir", type=str, default='/media/stuart/data')
parser.add_argument("--use_balance_train", type=bool, default=True)
args = parser.parse_args()
print(args)


class Train(object):
    def __init__(self, num_class, use_balance_train, use_cluster_loss, cluster_loss_weight, output_dir):
        """"""
        self.num_class = num_class
        self.use_balance_train = use_balance_train

        self.weights_save_dir = os.path.join(output_dir, 'weights')
        self.log_dir = os.path.join(output_dir, 'events')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.get_model(
            args.model_name,
            args.num_classes
        ).to(self.device).train()

        self.optimizer = self.get_optimizer(args.learning_rate)

        self.loss = self.get_loss(
            use_cluster_loss=use_cluster_loss,
            cluster_loss_weight=cluster_loss_weight
        )

        self.train_dataloader = self.get_dataloader(
            dataset_dir=args.dataset_root_dir,
            file_path=args.train_file if not use_balance_train else args.balance_train_file,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            is_train=True,
            use_balance_train=use_balance_train
        )

        self.val_dataloader = self.get_dataloader(
            dataset_dir=args.dataset_root_dir,
            file_path=args.val_file,
            batch_size=1,
            num_threads=args.num_threads,
            is_train=False
        )

        self.visualizer = self.get_visualizer(self.log_dir)

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

    def get_loss(self, use_cluster_loss, cluster_loss_weight):
        """"""
        loss = LossFactory(
            num_classes=self.num_class,
            use_cluster_loss=use_cluster_loss,
            cluster_loss_weight=cluster_loss_weight
        )

        return loss

    def get_dataloader(self, dataset_dir, file_path, batch_size=1, num_threads=1, is_train=True,
                       use_balance_train=False):
        """"""
        if not os.path.exists(dataset_dir):
            print("%s doesn't exist." % dataset_dir)
            return
        if not os.path.exists(file_path):
            print("%s doesn't exist." % file_path)
            return

        if use_balance_train:
            dataloader = ApolloBalanceTrainDataLoader(
                root_dir=dataset_dir,
                json_path=file_path,
                batch_size=batch_size
            )
        else:
            dataset = ApolloDataset(
                root_dir=dataset_dir,
                path_file=file_path,
                is_train=is_train
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_threads,
                pin_memory=True,
                drop_last=True if is_train else False
            )

        return dataloader

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

        print("Load from %s successfully." % ckpt_path)
        return iteration

    def save_checkpoint(self, iter):
        """"""
        ckpt_dir = os.path.join(self.weights_save_dir, self.model.__class__.__name__)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # save model weights, optimizer status, and train iteration
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iter
        }
        ckpt_path = os.path.join(ckpt_dir, 'balance_train_cluster_loss_iter_%d.pth' % iter)

        torch.save(save_dict, ckpt_path)
        print("Save checkpoint to %s" % ckpt_path)

    def run(self):
        iter = 0

        # load pretrained weights
        self.model.load_state_dict(
            torch.load('/home/stuart/.cache/torch/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'),
            strict=False
        )

        # resume train
        if args.resume:
            iter = self.load_checkpoint(args.resume)

        while True:
            for data in self.train_dataloader:

                if self.use_balance_train:
                    input, label = data[0], data[1]
                    input = torch.tensor(input, dtype=torch.float).to(self.device)
                    label = torch.tensor(label, dtype=torch.uint8).to(self.device)
                else:
                    input, label = data['image'].to(self.device), data['label'].to(self.device)

                # run forward
                self.optimizer.zero_grad()
                logits = self.model(input)['out']

                # compute loss, backward, update weights
                loss = self.loss.compute_loss(logits, label)
                loss['total_loss'].backward()
                self.optimizer.step()

                learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('iter %d: lr = %.5f, ' % (iter, learning_rate), loss)

                # visualize train process
                if iter % args.visualize_interval == 0:
                    self.visualizer.train_update(
                        iteration=iter,
                        learning_rate=learning_rate,
                        input=input.cpu(),
                        label=label.cpu(),
                        logits=logits.detach().cpu(),
                        loss=loss
                    )

                # save checkpoint
                if iter % args.save_interval == 0:
                    self.save_checkpoint(iter)

                iter += 1


if __name__ == '__main__':
    Train(
        num_class=args.num_classes,
        use_cluster_loss=args.use_cluster_loss,
        cluster_loss_weight=args.cluster_loss_weight,
        use_balance_train=args.use_balance_train,
        output_dir=args.output_dir
    ).run()
