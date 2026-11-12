import numpy as np

from .mean_square_error import MeanSquareError


class FixationMSEResponseCE(MeanSquareError):
    """Per-timestep loss chosen by the target row at each timestep.
    Targets have the same shape as MeanSquareError: (batch, timesteps, outputs).

    * all zeros       -> MSE towards zero        (fixation)
    * one-hot (sum 1) -> cross-entropy           (response)
    * all -1 (sum <0) -> ignored                 (padding)

    EventProp compiler only, with a 'Var' readout.
    """
    def set_target(self, genn_pop, y_true, shape, batch_size: int,
                   example_timesteps: int):
        # Targets are laid out exactly as for MeanSquareError
        super().set_target(genn_pop, y_true, shape, batch_size,
                           example_timesteps)
