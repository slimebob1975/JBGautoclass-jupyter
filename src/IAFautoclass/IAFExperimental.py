from skclean.models import Centroid
from skclean.detectors import PartitioningDetector, MCS
from sklearn.preprocessing import LabelEncoder

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
