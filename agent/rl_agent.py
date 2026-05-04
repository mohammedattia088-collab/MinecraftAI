"""
agent/rl_agent.py

Reinforcement Learning agent for MinecraftAI.
Implements Double DQN with Dueling Head architecture and Prioritized Experience Replay (PER).
Includes policy and target networks, epsilon-greedy exploration, and GPU/MPS support.

Author: MinecraftAI Senior Dev Team
Version: 1.0
"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import Optional, List, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from ..config import RL_PARAMS, TRAINING_PARAMS, MPS_DEVICE
    from ..utils.scheduler import EpsilonScheduler, LRScheduler
except ImportError:
    from OtherStuff.MinecraftAI.config import RL_PARAMS, TRAINING_PARAMS, MPS_DEVICE
    from OtherStuff.MinecraftAI.utils.scheduler import EpsilonScheduler, LRScheduler

# =========================
# Device configuration
# =========================
DEVICE = torch.device("mps" if MPS_DEVICE and torch.backends.mps.is_available() else "cpu")

# =========================
# Experience Replay (with optional Prioritized Replay)
# =========================
class PERMemory:
    """
    Prioritized Experience Replay buffer
    """
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        if capacity <= 0:
            raise ValueError("PER capacity must be positive")
        self.capacity: int = capacity
        self.alpha: float = alpha
        self.buffer: List = []
        self.priorities: List[float] = []
        self.pos: int = 0

    def push(self, transition: namedtuple, td_error: float = 1.0) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(td_error ** self.alpha)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = td_error ** self.alpha
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[namedtuple], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities, dtype=np.float64)
        else:
            prios = np.array(self.priorities[:len(self.buffer)], dtype=np.float64)

        total_priority = prios.sum()
        if total_priority <= 0 or not np.isfinite(total_priority):
            probs = np.full(len(self.buffer), 1.0 / len(self.buffer), dtype=np.float64)
        else:
            probs = prios / total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(td) + 1e-6) ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)

# =========================
# Dueling Double DQN Network
# =========================
class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture
    """
    def __init__(self, input_dim: int, n_actions: int) -> None:
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

# =========================
# Agent Class
# =========================
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class RLAgent:
    """
    Reinforcement Learning agent using Double DQN with Dueling Head and PER

    Args:
        state_dim (int): Dimension of the state space.
        n_actions (int): Number of possible actions.
        use_per (bool): Whether to use Prioritized Experience Replay.
        epsilon_scheduler (Optional[EpsilonScheduler]): Scheduler for epsilon decay.
        lr_scheduler (Optional[LRScheduler]): Scheduler for learning rate decay.
        logger (Optional[Logger]): Optional logger for debugging.
        frame_logger (Optional[Logger]): Optional logger for frame preprocessing metrics.
        tau (float): Soft update parameter for target network (default 1.0 for hard update).
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        use_per: bool = True,
        epsilon_scheduler: Optional[EpsilonScheduler] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        logger: Optional[Any] = None,
        frame_logger: Optional[Any] = None,
        tau: float = 1.0
    ) -> None:
        self.logger: Optional[Any] = logger
        self.frame_logger: Optional[Any] = frame_logger
        self.tau: float = tau

        # Dynamically check and enforce state_dim matches CNN output size
        # For this example, assume cnn_output_dim is obtained from RL_PARAMS or computed here
        cnn_output_dim = RL_PARAMS.get('cnn_output_dim', None)
        if cnn_output_dim is None:
            # If cnn_output_dim not provided, we try to infer from state_dim or log a warning
            if self.logger is not None:
                self.logger.log('warning', 'cnn_output_dim not specified in RL_PARAMS; assuming state_dim is correct', step=0)
            self.state_dim: int = state_dim
        else:
            if state_dim != cnn_output_dim:
                if self.logger is not None:
                    self.logger.log('warning', f"state_dim ({state_dim}) does not match cnn_output_dim ({cnn_output_dim}); adjusting state_dim to cnn_output_dim", step=0)
                self.state_dim = cnn_output_dim
            else:
                self.state_dim = state_dim

        self.n_actions: int = n_actions
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.use_per: bool = use_per
        self.epsilon_scheduler: Optional[EpsilonScheduler] = epsilon_scheduler
        self.lr_scheduler: Optional[LRScheduler] = lr_scheduler

        # Networks
        self.policy_net: DuelingDQN = DuelingDQN(self.state_dim, n_actions).to(DEVICE)
        self.target_net: DuelingDQN = DuelingDQN(self.state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=RL_PARAMS['learning_rate'])

        # Replay memory
        if use_per:
            self.memory: Union[PERMemory, deque] = PERMemory(RL_PARAMS['buffer_size'], alpha=RL_PARAMS['per_alpha'])
        else:
            self.memory = deque(maxlen=RL_PARAMS['buffer_size'])

        # Epsilon-greedy
        self.epsilon: float = TRAINING_PARAMS['epsilon_start']
        self.epsilon_min: float = TRAINING_PARAMS['epsilon_end']
        self.epsilon_decay: float = TRAINING_PARAMS['epsilon_decay']
        self.step_count: int = 0

        # Discount factor
        self.gamma: float = RL_PARAMS['gamma']

    # =========================
    # Action Selection
    # =========================
    async def select_action(self, state: Any) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (Any): Current state, can be numpy array or awaitable returning numpy array.

        Returns:
            int: Selected action.
        """
        self.step_count += 1
        # Await if state is awaitable (async environment)
        if hasattr(state, '__await__'):
            state = await state

        if state is None:
            if self.logger is not None:
                self.logger.log('warning', 'Received None state in select_action', step=self.step_count)
            return random.randrange(self.n_actions)

        if self.epsilon_scheduler is not None:
            self.epsilon = self.epsilon_scheduler.get_epsilon(self.step_count)
        if self.logger is not None:
            self.logger.log('epsilon', self.epsilon, step=self.step_count)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                if not isinstance(state, np.ndarray):
                    if self.logger is not None:
                        self.logger.log('warning', f"State is not np.ndarray in select_action: {type(state)}", step=self.step_count)
                    return random.randrange(self.n_actions)
                if state.shape != (self.state_dim,):
                    if self.logger is not None:
                        self.logger.log('warning', f"State shape mismatch in select_action: expected ({self.state_dim},) got {state.shape}", step=self.step_count)
                    return random.randrange(self.n_actions)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def decay_epsilon(self) -> None:
        """
        Decay epsilon value if no epsilon scheduler is provided.
        """
        if self.epsilon_scheduler is None:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    # =========================
    # Store experience
    # =========================
    async def store_experience(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> None:
        """
        Store experience in replay memory.

        Args:
            state (Any): Current state or awaitable returning state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (Any): Next state or awaitable returning next state.
            done (bool): Whether episode ended.
        """
        # Await if state or next_state are awaitable (async environment)
        if hasattr(state, '__await__'):
            state = await state
        if hasattr(next_state, '__await__'):
            next_state = await next_state

        if self.frame_logger is not None:
            self.frame_logger.log('store_experience', True, step=self.step_count)

        if state is None or next_state is None:
            if self.logger is not None:
                self.logger.log('warning', f"Received None state or next_state in store_experience at step {self.step_count}", step=self.step_count)
            return

        assert isinstance(state, np.ndarray), "State must be a numpy ndarray."
        if state.shape != (self.state_dim,):
            if self.logger is not None:
                self.logger.log('warning', f"State shape mismatch in store_experience: expected ({self.state_dim},) got {state.shape}", step=self.step_count)
            return
        assert isinstance(next_state, np.ndarray), "Next state must be a numpy ndarray."
        if next_state.shape != (self.state_dim,):
            if self.logger is not None:
                self.logger.log('warning', f"Next state shape mismatch in store_experience: expected ({self.state_dim},) got {next_state.shape}", step=self.step_count)
            return

        transition = Transition(state, action, reward, next_state, done)
        if self.use_per:
            self.memory.push(transition)
        else:
            self.memory.append(transition)

        if self.logger is not None:
            self.logger.log('reward', reward, step=self.step_count)

    # =========================
    # Update model
    # =========================
    async def update_model(
        self,
        batch_size: int = RL_PARAMS['batch_size'],
        beta: float = RL_PARAMS['per_beta_start']
    ) -> None:
        """
        Update the policy network using a batch of experiences.

        Args:
            batch_size (int): Number of samples to use for update.
            beta (float): Importance-sampling weight exponent for PER.
        """
        if len(self.memory) < batch_size:
            return

        if self.use_per:
            transitions, indices, weights = self.memory.sample(batch_size, beta)
        else:
            transitions = random.sample(self.memory, batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        batch = Transition(*zip(*transitions))
        states_np = np.array(batch.state)
        next_states_np = np.array(batch.next_state)

        # Safety checks
        if states_np is None or next_states_np is None:
            if self.logger is not None:
                self.logger.log('warning', 'Received None states or next_states in update_model', step=self.step_count)
            return
        if not isinstance(states_np, np.ndarray) or not isinstance(next_states_np, np.ndarray):
            if self.logger is not None:
                self.logger.log('warning', 'States or next_states not numpy arrays in update_model', step=self.step_count)
            return
        if states_np.shape[1] != self.state_dim:
            if self.logger is not None:
                self.logger.log('warning', f"States shape second dim must be {self.state_dim}, got {states_np.shape[1]}", step=self.step_count)
            return
        if next_states_np.shape[1] != self.state_dim:
            if self.logger is not None:
                self.logger.log('warning', f"Next states shape second dim must be {self.state_dim}, got {next_states_np.shape[1]}", step=self.step_count)
            return

        states = torch.tensor(states_np, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(next_states_np, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Next Q values for Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # TD error
        td_errors = q_target - q_values
        loss = (weights_tensor * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update PER priorities
        if self.use_per:
            td_errors_cpu = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_cpu)

        # Update epsilon
        if self.epsilon_scheduler is None:
            self.decay_epsilon()

        # Step LR scheduler if provided
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            if self.logger is not None:
                self.logger.log('learning_rate', self.lr_scheduler.get_lr(), step=self.step_count)

        # Log loss
        if self.logger is not None:
            self.logger.log('loss', loss.item(), step=self.step_count)

    # =========================
    # Update target network
    # =========================
    def update_target_network(self) -> None:
        """
        Update the target network parameters using soft update with factor tau.
        tau=1.0 corresponds to hard update (copy).
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
