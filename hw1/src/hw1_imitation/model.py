"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) 
            for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict action chunks."""
        x = self.input_layer(state)
        x = torch.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and true action chunks."""
        pred_chunk = self.forward(state)
        loss = nn.functional.mse_loss(pred_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        # The input of Flow Matching is the observation (state), the
        # action chunk at the current time (denoising) step, 
        # and the time step itself
        self.input_layer = nn.Linear(
            state_dim + action_dim * chunk_size + 1, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim * chunk_size)
        self.prng = torch.Generator(device="mps").manual_seed(17582)

    def forward(self, 
                state: torch.Tensor, 
                action_chunk_t: torch.Tensor, 
                tau: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict the velocity of the action chunk."""
        input = torch.cat(
            [
                state,
                action_chunk_t.flatten(start_dim=1),
                tau.reshape(1,1).repeat(state.shape[0], 1),
            ],
            dim=1
        )
        x = self.input_layer(input)
        x = torch.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def _make_initial_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Generate initial noise for the action chunk."""
        return torch.normal(
            mean=0, 
            std=1, 
            size=(state.shape[0], self.chunk_size, self.action_dim), 
            generator=self.prng,
            device=state.device
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss between predicted and true action chunks."""
        initial_noise = self._make_initial_noise(state)
        tau = torch.rand(size=(1,), 
                         generator=self.prng, 
                         device=state.device)
        action_chunk_t = tau * action_chunk + (1 - tau) * initial_noise
        pred_velocity = self.forward(state, action_chunk_t, tau)
        expected_velocity = (action_chunk - initial_noise)
        loss = nn.functional.mse_loss(pred_velocity, expected_velocity)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_chunk_t = self._make_initial_noise(state)
        for step in range(num_steps):
            tau = torch.tensor([(step + 1) / num_steps], device=state.device)
            action_chunk_t += (1 / num_steps) * self.forward(
                state, action_chunk_t, tau)

        return action_chunk_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
