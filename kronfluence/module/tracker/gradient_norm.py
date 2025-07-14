from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    GRADIENT_NORM_NAME,
)

class GradientNormTracker(BaseTracker):
    """Computes pairwise influence scores for a given module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)
        from kronfluence.module.linear import TrackedLinear
        assert isinstance(module, TrackedLinear), "GradientNormTracker can only be used with TrackedLinear right now."

    def register_hooks(self) -> None:
        """Sets up hooks to compute pairwise influence scores."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach()
            device = "cpu" if self.module.score_args.offload_activations_to_cpu else cached_activation.device
            cached_activation = cached_activation.to(
                device=device,
                dtype=self.module.score_args.per_sample_gradient_dtype,
                copy=True,
            )
            if self.module.factor_args.has_shared_parameters:
                raise NotImplementedError("Shared parameters are not supported for gradient norm computation.")
            self.cached_activations = cached_activation
            self.cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()

            assert isinstance(self.cached_activations, torch.Tensor), "Cached activations must be a tensor."
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.per_sample_gradient_dtype)
            cached_activation = self.cached_activations
            # Computes pairwise influence scores during backward pass.
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient,
                per_token=self.module.score_args.compute_per_token_scores,
            )
            del cached_activation, output_gradient
            if self.module.gradient_scale != 1.0:
                raise NotImplementedError("Gradient scale is not supported for gradient norm computation.")
        
            if self.module.score_args.compute_per_token_scores:
                per_sample_gradient_summed_squared = torch.einsum("btio,btio->bt", per_sample_gradient, per_sample_gradient)
            else:
                per_sample_gradient_summed_squared = torch.einsum("bio,bio->b", per_sample_gradient, per_sample_gradient)

            self.module.storage[GRADIENT_NORM_NAME] = torch.sqrt(per_sample_gradient_summed_squared)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    def finalize_iteration(self) -> None:
        """Clears all cached data from memory."""
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if pairwise score is available."""
        return self.module.storage[GRADIENT_NORM_NAME] is not None

    def accumulate_iterations(self) -> None:
        """Removes pairwise scores from memory after a single iteration."""
        self.release_memory()

    def release_memory(self) -> None:
        """Releases pairwise scores from memory."""
        self.clear_all_cache()
        self.module.storage[GRADIENT_NORM_NAME] = None
