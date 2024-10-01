from erasure.evaluations.manager import Evaluation
import time
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import defaultdict
import os

class RunTime():    
    def process(self, e: Evaluation):
        if not e.unlearned_model:
            start_time = time.time()

            e.unlearned_model = e.unlearner.unlearn()
            metric_value = time.time() - start_time

        e.add_value('RunTime', metric_value)

        return e
    
class Accuracy():
    def process(self, e: Evaluation): 
        
        model1 = e.unlearner.model
        model2 = e.unlearned_model

        print(type(model1))
        print(type(model2))

        test_loader, _ = e.unlearner.dataset.get_loader_for('test')

        og_accuracy = self.compute_accuracy(test_loader, model1.model)
        new_accuracy = self.compute_accuracy(test_loader,model2.model)

        print("ORIGINAL ACCURACY WAS ", og_accuracy)
        print("NEW ACCURACY IS ", new_accuracy)

        e.add_value('Accuracies', {'Original_accuracy:':og_accuracy, 'New_accuracy:':new_accuracy})

        return e


    def compute_accuracy(self, test_loader, model):
        
        var_labels, var_preds = [], [],
        with torch.no_grad():
            for batch, (X, labels) in enumerate(test_loader):

                _,pred = model(X.to(model.device))

                var_labels += list(labels.squeeze().to('cpu').numpy())
                var_preds += list(pred.squeeze().to('cpu').numpy())

            accuracy = self.accuracy(var_labels, var_preds)

        return accuracy

    def accuracy(self, testy, probs):
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc
    
class ForgetSetInfo():
    def process(self, e:Evaluation):
        e.add_value('Size of identified forget set', len(e.forget_set))
        
        forget_set_loader = e.unlearner.dataset.get_loader_for_ids(e.forget_set)

        distributions = defaultdict(int)

        for _,labels in forget_set_loader:
            for l in labels:
                distributions[l.item()] += 1

        distributions = {key:(value/len(e.forget_set)) for key,value in distributions.items()}

        e.add_value('Distribution of classes in the forget set', distributions)

        return e


    
class SaveValues():
    def __init__(self, path):
        self.path = path

    def process(self, e:Evaluation):

        with open(self.path, 'a') as json_file:
            json.dump(e.data_info, json_file, indent=4)
            json_file.write(',')




