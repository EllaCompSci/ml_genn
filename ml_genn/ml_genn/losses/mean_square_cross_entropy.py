from .loss import Loss


class MeanSquareErrorCrossEntropy(Loss):
    # mean_square_error_cross_entropy.py
    def __init__(self, fixation_weight: float = 1.0, record_key=None):
        super().__init__(record_key)
        self.fixation_weight = fixation_weight
    @property
    def ground_truth(self) -> str:
        return "label_and_fixation"