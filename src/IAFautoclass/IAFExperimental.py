from random import Random
import numpy as np
from skclean.models import RobustLR, Centroid
from skclean.detectors import PartitioningDetector, MCS, InstanceHardness, RandomForestDetector
from sklearn.preprocessing import LabelEncoder

# Models
class IAFRobustLogisticRegression(RobustLR):
    def __init__(self, PN=.2, NP=.2, C=np.inf, max_iter=4000, random_state=None):
        super().__init__(PN=.2, NP=.2, C=np.inf, max_iter=4000, random_state=None)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y, sample_weight=None):
        self.label_encoder = self.label_encoder.fit(y)
        super().fit(X, self.label_encoder.transform(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(super().predict(X))

class IAFRobustCentroid(Centroid):
    
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        super().fit(X, self.label_encoder.transform(y))
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(super().predict(X))


# Detectors
class LabelEncodedDetector():
    def __init__(self):
        self.label_encoder = LabelEncoder()

class IAFPartitioningDetector(PartitioningDetector):

    def __init__(self, classifier=None, n_partitions=5, n_jobs=1, random_state=None):
        super().__init__(classifier=classifier, n_partitions=n_partitions, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))


class IAFMCS(MCS):

    def __init__(self, classifier=None, n_steps=20, n_jobs=1, random_state=None):
        super().__init__(classifier=classifier, n_steps=n_steps, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))

class IAFInstanceHardness(InstanceHardness):
    
    def __init__(self, classifiers=None, cv=None, n_jobs=1, random_state=None):
        super().__init__(classifiers=classifiers, cv=cv, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))
    

class IAFRandomForestDetector(RandomForestDetector):
    # TODO: method får inte ges till init av RFD
    def __init__(self, method='bootstrap', n_estimators=101, sampling_ratio=None, cv=None, n_jobs=1, random_state=None):
        super().__init__(method=method, n_estimators=n_estimators, sampling_ratio=sampling_ratio, 
                        cv=cv, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))