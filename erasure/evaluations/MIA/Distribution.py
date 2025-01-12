import numpy as np
import numpy.linalg
import scipy
import sklearn

from erasure.core.base import Configurable


class Distribution(Configurable):

    def init(self):
        X, y = self.local.dataset.partitions["all"][:]
        self.distribution_out = X[y == 0].squeeze()
        self.distribution_in  = X[y == 1].squeeze()

        print(len(self.distribution_in))

        try:
            self.curve_out = scipy.stats.gaussian_kde(self.distribution_out)
            self.curve_in  = scipy.stats.gaussian_kde(self.distribution_in)
        except (numpy.linalg.LinAlgError, ValueError):
            self.curve_out = None
            self.curve_in = None


    def evaluate(self, z):
        """ Returns the probability of a sample z to be in both the distribution (in & out) """

        if self.curve_out and self.curve_in:
            return (
                self.curve_out.evaluate(z),
                self.curve_in.evaluate(z)
            )








