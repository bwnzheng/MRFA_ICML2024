import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_manager import DummyDataset, AugmentMemoryDataset
from models.mrfa import MRFA

EPSILON = 1e-8


init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 70
lrate = 0.1
milestones = [30, 50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2


class Replaymrfa(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        self.save_task_checkpoints = args.get('save_task_checkpoints')
        self.load_ckpt = args.get('load_ckpt')

        self.perturb_p = np.array(args['perturb_p'])
        self.disable_perturb = args.get('disable_perturb', True)
        self.num_augmem = args.get('num_augmem', 1)
        self.perturb_all = args.get('perturb_all', False)

        self.MRFA = MRFA()

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        if self.save_task_checkpoints and self._cur_task in self.save_task_checkpoints:
            self.save_checkpoint(f'checkpoints/{self.args["convnet_type"]}_{self.args["dataset"]}_{self.args["init_cls"]}-{self.args["increment"]}_{self.args["model_name"]}')

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "memory": self._get_memory(), # memory for next task
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        model_state_dict = ckpt['model_state_dict']
        self._network.load_state_dict(model_state_dict)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.task_train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
        )

        # Loader
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self.load_ckpt is not None and self._cur_task < len(self.load_ckpt):
            self.load_checkpoint(self.load_ckpt[self._cur_task])
            return

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                self._network.train()
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))

        base_dataset = self.task_train_dataset
        base_num_samples = len(base_dataset)

        aug_imgs, aug_targets = self._repeat_memory()
        mem_aug_dataset = AugmentMemoryDataset(aug_imgs, aug_targets, base_dataset.trsf, index_offset=base_num_samples, use_path=base_dataset.use_path)

        dataset_list = [base_dataset, *([mem_aug_dataset]*self.num_augmem)]
        concat_dataset = ConcatDataset(dataset_list)
        train_loader = DataLoader(
            concat_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.MRFA.register_perturb_forward_prehook(self._network, self.args['convnet_type'])

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (indices, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                if (((perturb_indices := indices - base_num_samples) >= 0).any() or self.perturb_all) and not self.disable_perturb:
                    perturb_mask = perturb_indices >= 0 if not self.perturb_all else indices >= 0
                    perturb_indices = perturb_indices[perturb_mask]
                    self.MRFA.feature_augmentation(self._network, inputs[perturb_mask], targets[perturb_mask], self.args['convnet_type'])

                    self.MRFA.perturbation_idices.extend(np.arange(len(perturb_indices)).tolist())

                    self.MRFA.perturbation_idices_inbatch.extend(perturb_mask.nonzero().flatten().tolist())
                    self.MRFA.perturbation_layers.extend(np.random.randint(0, len(self.perturb_p), len(perturb_indices)).tolist())
                    self.MRFA.perturbation_factor = (self.perturb_p[self.MRFA.perturbation_layers] * np.random.rand(len(perturb_indices))).tolist()

                self._network.train()
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                self.MRFA._init_inbatch_properties()

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        if len(self.MRFA.remove_handles) > 0:
            for handle in self.MRFA.remove_handles:
                handle.remove()
            self.MRFA.remove_handles.clear()

        logging.info(info)

    def _augment_memory_features(self):
        memory_images, memory_targets = self._get_memory()
        memory_dataset = DummyDataset(memory_images, memory_targets, self.test_dataset.trsf)
        memory_loader = DataLoader(memory_dataset, len(memory_dataset), shuffle=False, num_workers=num_workers)
        _, memory_samples, memory_targets = next(iter(memory_loader))
        
        self.MRFA.feature_augmentation(self._network, memory_samples.to(self._device), memory_targets.to(self._device), self.args['convnet_type'])
    
    def _repeat_memory(self):
        memory_images, memory_targets = self._get_memory()

        return memory_images, memory_targets