import json
from collections import defaultdict

from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.utils import compute_accuracy


class Accuracy(Measure):
    def process(self, e: Evaluation): 
        
        model1 = e.unlearner.predictor
        model2 = e.unlearned_model

        test_loader, _ = e.unlearner.dataset.get_loader_for('test')

        og_accuracy = compute_accuracy(test_loader, model1.model)
        new_accuracy = compute_accuracy(test_loader,model2.model)

        print("ORIGINAL ACCURACY WAS ", og_accuracy)
        print("NEW ACCURACY IS ", new_accuracy)

        e.add_value('Accuracies', {'Original_accuracy:':og_accuracy, 'New_accuracy:':new_accuracy})

        return e


class AUS(Measure):
    """ Adaptive Unlearning Score """

    def process(self, e: Evaluation):
        or_model = e.unlearner.predictor
        ul_model = e.unlearned_model

        test_loader, _ = e.unlearner.dataset.get_loader_for('test')
        forget_loader, _ = e.unlearner.dataset.get_loader_for('forget set')

        or_test_accuracy = compute_accuracy(test_loader, or_model.model)
        ul_test_accuracy = compute_accuracy(test_loader, ul_model.model)
        ul_forget_accuracy = compute_accuracy(forget_loader, ul_model.model)

        aus = (1 - (or_test_accuracy - ul_test_accuracy)) / (1 + abs(ul_test_accuracy - ul_forget_accuracy))

        print("Adaptive Unlearning Score:", aus)
        e.add_value("AUS", aus)

        return e


class ForgetSetInfo(Accuracy):
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


class SaveValues(Accuracy):
    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.path = self.params['path']

    def process(self, e:Evaluation):

        with open(self.path, 'a') as json_file:
            json.dump(e.data_info, json_file, indent=4)
            json_file.write(',')

        return e

