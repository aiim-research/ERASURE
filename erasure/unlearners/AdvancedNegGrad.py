from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction

from erasure.utils.cfg_utils import init_dflts_to_of

class AdvancedNegGrad(TorchUnlearner):
    def init(self):
        """
        Initializes the AdvancedNegGrad class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']  
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget'] 

        self.predictor.optimizer = self.local.config['parameters']['optimizer']
        module_name, class_name = self.predictor.optimizer["class"].rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        optimizer_class = getattr(module, class_name)
        self.predictor.optimizer = optimizer_class(self.predictor.model.parameters(), **self.predictor.optimizer["parameters"])

    def __unlearn__(self):
        """
        An implementation of the Advanced NegGrad unlearning algorithm proposed in the following paper:
        "Choi, D. and Na, D., 2023. Towards machine unlearning benchmarks: Forgetting the personal identities in facial recognition systems. arXiv preprint arXiv:2311.02240."
        
        Codebase taken from the original implementation: https://github.com/ndb796/MachineUnlearning
        """

        self.info(f'Starting AdvancedNegGrad with {self.epochs} epochs')

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction('0'))

        dataloader_iterator = iter(forget_loader)

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            for X_retain, labels_retain in retain_loader:
                X_retain, labels_retain = X_retain.to(self.device), labels_retain.to(self.device)
                
                self.predictor.optimizer.zero_grad() 

                try: 
                    (X_forget, labels_forget) = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(forget_loader)
                    (X_forget, labels_forget) = next(dataloader_iterator)
                
                if X_forget.size(0) != X_forget.size(0):
                    continue

                _, output_retain = self.predictor.model(X_retain.to(self.device))
                _, output_forget = self.predictor.model(X_forget.to(self.device))
                
                loss_ascent_forget = -self.predictor.loss_fn(output_forget, labels_forget.to(self.device))
                loss_retain = self.predictor.loss_fn(output_retain, labels_retain.to(self.device))

                # Overall loss
                joint_loss = loss_ascent_forget + loss_retain

                losses.append(joint_loss.to('cpu').detach().numpy())

                joint_loss.backward()
                self.predictor.optimizer.step()
            
            epoch_loss = sum(losses) / len(losses)
            self.info(f'AdvancedNegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget

        init_dflts_to_of(self.local.config, 'optimizers', 'torch.optim.Adam') # Default optimizer is Adam