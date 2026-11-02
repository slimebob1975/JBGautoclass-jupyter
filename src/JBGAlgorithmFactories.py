"""Estimator construction methods for Algorithm enum members.

Keep Algorithm and its enum values in JBGMeta for serialization compatibility.
The mixin is deliberately not an Enum and introduces no new enum members.
"""
from __future__ import annotations

from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from imblearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from JBGFeatureSelection import NonEmptySelectFromModel
from JBGTransformers import NNClassifier3PL, MLPKerasClassifier
from JBGJaxTransformers import FlaxClassifier


class AlgorithmEstimatorFactories:
    """Estimator factory implementation inherited by JBGMeta.Algorithm."""

    def do_DUMY(self, max_iterations: int, size: int)-> DummyClassifier:
        return DummyClassifier()

    def do_SRF1(self, max_iterations: int, size: int)-> StackingClassifier:
        estimators = [ \
                ('for', RandomForestClassifier()),\
                ('bfor',BalancedRandomForestClassifier())\
                ]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    def do_BARF(self, max_iterations: int, size: int)-> BalancedRandomForestClassifier:
        return BalancedRandomForestClassifier()

    def do_BABC(self, max_iterations: int, size: int)-> BalancedBaggingClassifier:
        return BalancedBaggingClassifier()

    def do_RUBC(self, max_iterations: int, size: int)-> RUSBoostClassifier:
        return RUSBoostClassifier()

    def do_EAEC(self, max_iterations: int, size: int)-> EasyEnsembleClassifier:
        return EasyEnsembleClassifier()

    def do_LRN(self, max_iterations: int, size: int)-> LogisticRegression:
        return LogisticRegression(max_iter=max_iterations)

    def do_KNN(self, max_iterations: int, size: int)-> KNeighborsClassifier:
        return KNeighborsClassifier()

    def do_RADN(self, max_iterations: int, size: int)-> RadiusNeighborsClassifier:
        return RadiusNeighborsClassifier(outlier_label="most_frequent")

    def do_DTC(self, max_iterations: int, size: int)-> DecisionTreeClassifier:
        return DecisionTreeClassifier()

    def do_GNB(self, max_iterations: int, size: int)-> GaussianNB:
        return GaussianNB()

    def do_MNB(self, max_iterations: int, size: int)-> MultinomialNB:
        return MultinomialNB()

    def do_BNB(self, max_iterations: int, size: int)-> BernoulliNB:
        return BernoulliNB()

    def do_CNB(self, max_iterations: int, size: int)-> ComplementNB:
        return ComplementNB()

    def do_REC(self, max_iterations: int, size: int)-> RidgeClassifier:
        return RidgeClassifier(max_iter=max_iterations)

    def do_PCN(self, max_iterations: int, size: int)-> Perceptron:
        return Perceptron(max_iter=max_iterations)

    def do_PAC(self, max_iterations: int, size: int)-> SGDClassifier:
        return SGDClassifier(
            loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
            max_iter=max_iterations
        )

    def do_CBST(self, max_iterations: int, size: int):
        from JBGNativeCatBoost import OrdinaryNativeCatBoostClassifier
        return OrdinaryNativeCatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                        class_weight="None", random_seed=42)

    def do_RFCL(self, max_iterations: int, size: int)-> RandomForestClassifier:
        return RandomForestClassifier()

    def do_LSVC(self, max_iterations: int, size: int)-> LinearSVC:
        return LinearSVC(max_iter=max_iterations)

    def do_SLSV(self, max_iterations: int, size: int)-> Pipeline:
        return Pipeline([
            ('feature_selection', NonEmptySelectFromModel(LinearSVC(max_iter=max_iterations, penalty="l1", dual=False))),
            ('classification', LinearSVC(max_iter=max_iterations, penalty="l2", dual=True))])

    def do_SVC(self, max_iterations: int, size: int):
        if size < self.limit:
            # Keep probability estimates available for multiclass ROC-AUC scoring and
            # downstream probability-based diagnostics. Binary AUC may still use the
            # cheaper decision_function path through the scoring resolver.
            return SVC(max_iter=max_iterations, probability=True)

        # TODO: communicate with terminal to warn?
        # print("\nNotice: SVC model was exchange for LinearSVC since n_samples > {0}\n".format(self.LIMIT_SVC))
        return self.do_LSVC(max_iterations=max_iterations, size=size)

    def do_STCL(self, max_iterations: int, size: int) -> SelfTrainingClassifier:
        return SelfTrainingClassifier(self.do_SVC(max_iterations, size))

    def do_SGDE(self, max_iterations: int, size: int)-> SGDClassifier:
        return SGDClassifier(max_iter=max_iterations)

    def do_NCT(self, max_iterations: int, size: int)-> NearestCentroid:
        return NearestCentroid()

    def do_LDA(self, max_iterations: int, size: int)-> LinearDiscriminantAnalysis:
        return LinearDiscriminantAnalysis()

    def do_QDA(self, max_iterations: int, size: int)-> QuadraticDiscriminantAnalysis:
        # The unregularized sklearn default (reg_param=0.0) raises LinAlgError
        # for rank-deficient class covariance matrices. Start spot-checking at
        # the same smallest regularization value already used by the QDA grid.
        from JBGMeta import AlgorithmGridSearchParams
        reg_params = AlgorithmGridSearchParams.QDA.parameters["reg_param"]
        return QuadraticDiscriminantAnalysis(reg_param=min(reg_params))

    def do_BGC(self, max_iterations: int, size: int)-> BaggingClassifier:
        return BaggingClassifier()

    def do_ETC(self, max_iterations: int, size: int)-> ExtraTreesClassifier:
        return ExtraTreesClassifier()

    def do_ABC(self, max_iterations: int, size: int)-> AdaBoostClassifier:
        return AdaBoostClassifier()

    def do_GBC(self, max_iterations: int, size: int)-> GradientBoostingClassifier:
        return GradientBoostingClassifier()

    def do_HIST(self, max_iterations: int, size: int)-> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier()

    def do_MLPC(self, max_iterations: int, size: int)-> MLPClassifier:
        return MLPClassifier(max_iter=max_iterations)

    def do_MLP2(self, max_iterations: int, size: int)-> MLPClassifier:
        # Keep the "Light" variant light through its smaller search grid, not
        # by silently overriding the configured iteration budget.
        return MLPClassifier(max_iter=max_iterations)

    def do_GPC(self, max_iterations: int, size: int)-> GaussianProcessClassifier:
        return GaussianProcessClassifier(max_iter_predict=max_iterations)

    def do_VOTG(self, max_iterations: int, size: int) -> Filter:
        clf1 = LinearSVC()
        clf2 = RandomForestClassifier()
        clf3 = LogisticRegression()
        estimators=[('lsvc', clf1), ('rfcl', clf2), ('lrn', clf3)]
        return VotingClassifier(estimators=estimators)

    def do_TORA(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return self.call_PYNN(activation='relu', optimizer='adam')

    def do_TORS(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return self.call_PYNN(activation='relu', optimizer='sgd')

    def do_TOTA(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return self.call_PYNN(activation='tanh', optimizer='adam')

    def do_TOTS(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return self.call_PYNN(activation='tanh', optimizer='sgd')

    def do_TOSA(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return self.call_PYNN(activation='sigmoid', optimizer='adam')

    def do_TOSS(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return self.call_PYNN(activation='sigmoid', optimizer='sgd')

    def call_PYNN(self, activation: str, optimizer: str)-> NNClassifier3PL:
        shallow_sigmoid_sgd = activation == "sigmoid" and optimizer == "sgd"
        rate = 0.1 if shallow_sigmoid_sgd else (0.001 if optimizer == "adam" else 0.02)
        depth = 0 if shallow_sigmoid_sgd else 2
        return NNClassifier3PL(activation=activation, optimizer=optimizer,
                               learning_rate=rate, num_hidden_layers=depth, max_epochs=50,
                               verbose=False, train_split=True)

    def do_KERA(self, max_iterations: int, size: int)-> NNClassifier3PL:
        return  MLPKerasClassifier()

    def do_FUTV(self, max_iterations: int, size: int) -> Filter:
        clf1 = MLPClassifier(max_iter=max_iterations)
        clf2 = RandomForestClassifier()
        clf3 = AdaBoostClassifier()
        estimators=[('mlpc', clf1), ('rfcl', clf2), ('abc', clf3)]
        return VotingClassifier(estimators=estimators, voting = 'soft')

    def do_FUTS(self, max_iterations: int, size: int) -> Filter:
        clf1 = MLPClassifier(max_iter=max_iterations)
        clf2 = RandomForestClassifier()
        clf3 = AdaBoostClassifier()
        estimators=[('mlpc', clf1), ('rfcl', clf2), ('abc', clf3)]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    def do_FLAX(self, max_iterations: int, size: int)-> FlaxClassifier:
        return  FlaxClassifier(verbose=False)
