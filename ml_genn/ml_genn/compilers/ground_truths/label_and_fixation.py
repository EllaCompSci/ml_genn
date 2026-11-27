import numpy as np

from .ground_truth import GroundTruth
from ml_genn.utils.model import NeuronModel
from pygenn import VarAccess
from typing import List, Tuple


class LabelAndFixation(GroundTruth):
    """Ground truth for the combined fixation-MSE / response-SCE loss.

    Targets are passed as per-timestep integer labels, shape
    ``(batch, example_timesteps)``:

    * ``0``      fixation (output voltage is pushed towards 0 with MSE)
    * ``1..C``   response, class ``c - 1`` (one class per trial)
    * ``-1``     padding (no loss)

    On device this becomes:

    * ``YTrue`` / ``YTrueBack``: the trial's response class, used by the
      trial-level SCE on the AvgVarExpWeight readout (as ExampleLabel)
    * ``Fixation``: a per-example, per-timestep 0/1 mask used by the
      forward pass to decide when to write an MSE error into the ring
    """
    def add_to_neuron(self, backward: bool, model: NeuronModel,
                      shape, batch_size: int, example_timesteps: int):
        # Trial label, exactly as ExampleLabel
        model.add_var("YTrue", "uint16_t", 0,
                      VarAccess.READ_ONLY_SHARED_NEURON, reset=False)
        if backward:
            model.add_var("YTrueBack", "uint16_t", 0,
                          VarAccess.READ_ONLY_SHARED_NEURON, reset=False)

        # Fixation mask, shared across neurons: (batch, timestep)
        model.add_egp("Fixation", "uint8_t*",
                      np.zeros(batch_size * example_timesteps, dtype=np.uint8))

    def push_to_device(self, genn_pop, y_true, shape, batch_size: int,
                       example_timesteps: int):
        y_true = np.asarray(y_true)
        if (y_true.ndim != 2 or y_true.shape[1] != example_timesteps
                or len(y_true) > batch_size):
            raise RuntimeError(f"LabelAndFixation ground truth should be "
                               f"(<= {batch_size}, {example_timesteps}) "
                               f"integer labels, got {y_true.shape}")

        # Trial label = the response class (labels are 1-based, 0 = fixation)
        response = np.where(y_true > 0, y_true, 0)
        label = response.max(axis=1) - 1
        if np.any(label < 0):
            raise RuntimeError("LabelAndFixation: a trial has no response "
                               "timesteps, so it has no class label")
        # Every response timestep in a trial must agree on the class
        mixed = np.any((y_true > 0) & (y_true != (label + 1)[:, None]), axis=1)
        if np.any(mixed):
            raise RuntimeError(f"LabelAndFixation: trials "
                               f"{np.flatnonzero(mixed)} contain more than one "
                               f"response class - this loss assumes one per trial")

        # Write label (same layout as ExampleLabel)
        y_var = genn_pop.vars["YTrue"]
        if batch_size == 1:
            y_var.view[:] = label[:]
        else:
            y_var.view[:len(label), 0] = label
        y_var.push_to_device()

        # Write fixation mask; unused batch slots stay 0 (no MSE)
        fix_egp = genn_pop.extra_global_params["Fixation"]
        fix_egp.view[:] = 0
        fix_egp.view[:y_true.size] = (y_true == 0).astype(np.uint8).flatten()
        fix_egp.push_to_device()

    @property
    def backward_shared_neuron_var_reset(self) -> List[Tuple[str, str, str]]:
        # Copy YTrue into YTrueBack at the end of each trial, as ExampleLabel
        return [("YTrueBack", "uint16_t", "YTrue")]
