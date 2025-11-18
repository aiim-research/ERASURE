from erasure.unlearners.torchunlearner import TorchUnlearner

import torch
import torch.nn.functional as F

import numpy as np
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs

class BadTeaching(TorchUnlearner):
    def init(self):
        """
        Initializes the Bad Teaching class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.transform = self.local.config['parameters']['transform']
        self.batch_size = self.dataset.batch_size
        self.KL_temperature = self.local.config['parameters']['KL_temperature']

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})

        self.cfg_bad_teacher = self.local.config['parameters']['bad_teacher']
        self.current_bt = Local(self.cfg_bad_teacher)
        self.current_bt.dataset = self.dataset

    def UnlearnerLoss(self, output, labels, gt_logits, bt_logits, KL_temperature):
        labels = torch.unsqueeze(labels, dim = 1)
        
        gt_output = F.softmax(gt_logits / KL_temperature, dim=1)
        bt_output = F.softmax(bt_logits / KL_temperature, dim=1)

        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = labels * bt_output + (1-labels)*gt_output
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

    def unlearning_step(self, model, good_teacher, bad_teacher, unlearn_data_loader, optimizer, 
                device, KL_temperature):
        losses = []
        for batch in unlearn_data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, gt_logits = good_teacher(x)
                _, bt_logits = bad_teacher(x)
            _, output = model(x)
            optimizer.zero_grad()
            loss = self.UnlearnerLoss(output = output, labels=y, gt_logits=gt_logits,
                                      bt_logits=bt_logits, KL_temperature=KL_temperature)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        return np.mean(losses)

    def __unlearn__(self):
        """
        An implementation of the Bad Teaching unlearning algorithm proposed in the following paper:
        "Chundawat, V.S., Tarun, A.K., Mandal, M. and Kankanhalli, M., 2023, June. Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7210-7217)."
        
        Codebase taken from the original implementation: https://github.com/vikram2000b/bad-teaching-unlearning
        """

        self.info(f'Starting BadTeaching with {self.epochs} epochs')

        self.bad_teacher = self.global_ctx.factory.get_object(self.current_bt)
        
        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain)
        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget)

        self.retain_set = retain_loader.dataset
        self.forget_set = forget_loader.dataset
        
        unlearning_data = UnLearningData(forget_data=self.forget_set, retain_data=self.retain_set, transform=self.transform)
        if isinstance(self.retain_set[0][0],Data):
            unlearning_loader = GeometricDataLoader(unlearning_data, batch_size=self.batch_size, shuffle=True)
        else:
            unlearning_loader = DataLoader(unlearning_data, batch_size = self.batch_size, shuffle=True)

        self.info(f'Number of steps per epoch: { len(unlearning_loader)}')

        good_teacher = copy.deepcopy(self.predictor.model)

        good_teacher.eval()
        self.bad_teacher.model.eval()        

        for epoch in range(self.epochs):
            loss = self.unlearning_step(model = self.predictor.model, good_teacher= good_teacher, 
                            bad_teacher=self.bad_teacher.model, unlearn_data_loader=unlearning_loader, 
                            optimizer=self.predictor.optimizer, device=self.device, KL_temperature=self.KL_temperature)
            self.info(f'Epoch {epoch} Unlearning Loss {loss}')
            
        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['transform'] = self.local.config['parameters'].get("transform", None) # Default transformation applied to the data is None
        self.local.config['parameters']['KL_temperature'] = self.local.config['parameters'].get("KL_temperature", 1.0) # Default KL temperature is 1.0
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam

        if 'bad_teacher' not in self.local.config['parameters']: 
            self.local.config['parameters']['bad_teacher'] = copy.deepcopy(self.global_ctx.config.predictor) # Default bad teacher has the same configuration of the original predictor trained for 0 epochs
            self.local.config['parameters']['bad_teacher']['parameters']['cached'] = False
            self.local.config['parameters']['bad_teacher']['parameters']['epochs'] = 0
            self.local.config['parameters']['bad_teacher']['parameters']['training_set'] = self.local.config['parameters']['ref_data_retain']
        else: 
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['predictor']['parameters'].get('training_set',self.local.config['parameters']['ref_data_retain'])

        self.local.config['parameters']['bad_teacher']['parameters']['cached'] = self.local.config['parameters']['bad_teacher']['parameters'].get('cached',False)

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
            if isinstance(self.forget_data[index], dict):
                x = self.forget_data[index].values()
                x = self.transform(list(x)) if self.transform else list(x)
                x = torch.tensor(x)
            else:
                x = self.transform(self.forget_data[index][0]) if self.transform else self.forget_data[index][0]
            y = 1
            return x,y
        else:
            if isinstance(self.retain_data[index - self.forget_len], dict):
                x = self.retain_data[index - self.forget_len].values()
                x = self.transform(list(x)) if self.transform else list(x)
                print(x)
                x = torch.tensor(x)
            else:
                x = self.transform(self.retain_data[index - self.forget_len][0]) if self.transform else self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y
