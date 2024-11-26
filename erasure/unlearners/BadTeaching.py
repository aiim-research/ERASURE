from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from fractions import Fraction

import torch
import torch.nn.functional as F

import numpy as np
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class BadTeaching(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the Bad Teaching class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """

        super().__init__(global_ctx, local_ctx)

    def UnlearnerLoss(self, output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
        labels = torch.unsqueeze(labels, dim = 1)
        
        f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

    def unlearning_step(self, model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
                device, KL_temperature):
        losses = []
        for batch in unlearn_data_loader:
            x, y = batch
            # remove second dimension - only for mucac 
            # x = x.squeeze(1)
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, full_teacher_logits = full_trained_teacher(x)
                _, unlearn_teacher_logits = unlearning_teacher(x)
            _, output = model(x)
            optimizer.zero_grad()
            loss = self.UnlearnerLoss(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
                    unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        return np.mean(losses)

    def __unlearn__(self):
        """
        
        """

        self.global_ctx.logger.info(f'Starting BadTeaching with {self.epochs} epochs')

        self.retain_set = self.dataset.get_dataset_from_partition(self.ref_data_retain)
        self.forget_set = self.dataset.get_dataset_from_partition(self.ref_data_forget)

        retain_data = torch.stack([x[0] for x in self.retain_set])
        forget_data = torch.stack([x[0] for x in self.forget_set])

        unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data, transform=self.transform)
        unlearning_loader = DataLoader(unlearning_data, batch_size = self.batch_size, shuffle=True)

        print("Number of steps per epoch: ", len(unlearning_loader))

        good_teacher = copy.deepcopy(self.predictor.model)
        full_trained_teacher = None #TODO 

        good_teacher.eval()
        full_trained_teacher.eval()        

        for epoch in range(self.epochs):
            loss = self.unlearning_step(model = self.predictor.model, unlearning_teacher= good_teacher, 
                            full_trained_teacher=full_trained_teacher, unlearn_data_loader=unlearning_loader, 
                            optimizer=self.optimizer, device=self.device, KL_temperature=self.KL_temperature)
            print("Epoch {} Unlearning Loss {}".format(epoch, loss))
            
        return self.predictor

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data, transform):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.transform = transform
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.transform(self.forget_data[index]) if self.transform else self.forget_data[index]
            y = 1
            return x,y
        else:
            x = self.transform(self.retain_data[index - self.forget_len]) if self.transform else self.retain_data[index - self.forget_len]
            y = 0
            return x,y

def check_configuration(self):
        super().check_configuration()

        self.epochs = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.ref_data_retain = self.local.config['parameters'].get("ref_data_retain", 'retain set')  # Default reference data is retain
        self.ref_data_forget = self.local.config['parameters'].get("ref_data_forget", 'forget set')  # Default reference data is forget

        self.transform = self.local.config['parameters'].get("transform", None)
        self.batch_size = self.local.config['parameters'].get("batch_size", 64)
        self.KL_temperature = self.local.config['parameters'].get("KL_temperature", 1.0)

        optimizer = self.local.config['parameters'].get("optimizer", "adam")
        if optimizer == "adam":
            optimizer = {
                "class": "torch.optim.Adam",
                "parameters": {
                    "lr": 0.001
                }
            }
        module_name, class_name = optimizer["class"].rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        optimizer_class = getattr(module, class_name)
        self.optimizer = optimizer_class(self.predictor.model.parameters(), **optimizer["parameters"])