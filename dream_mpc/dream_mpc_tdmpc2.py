import torch
from torch import nn
from torch.func import grad, vmap
import torch.nn.functional as F

from common import math
from common.planner import FreezeParameters
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion, bmpc_tdmpc2_model_conversion
from tensordict import TensorDict

from scipy.optimize import linear_sum_assignment

class DreamMPC_TDMPC2(torch.nn.Module):
	"""Dream-MPC (TD-MPC2) agent. Implements training + inference.
	Can be used for single-task experiments (multi-task not tested),
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self._prev_best_actions = torch.zeros(
				self.cfg.horizon,
				self.cfg.action_dim,
				device=self.device,
			)
		self._prev_elite_actions = torch.zeros(
			self.cfg.horizon,
			self.cfg.elite_k,
			self.cfg.action_dim,
			device=self.device,
		)
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=torch.get_default_device())
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		state_dict = bmpc_tdmpc2_model_conversion(state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu(), None
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu(), None

	def _estimate_value(self, z, actions, task, regularization_coefficient=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		uncertainties = torch.empty_like(actions, device=self.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * reward
			uncertainty = self._estimate_uncertainty(z, actions[t], task)
			uncertainties[t, :] = uncertainty
			if regularization_coefficient:
				G -= regularization_coefficient * uncertainty
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		action, _ = self.model.pi(z, task)
		val = G + discount * self.model.Q(z, action, task, return_type='avg')

		uncertainty = self._estimate_uncertainty(z, action, task)
		uncertainties[-1, :] = uncertainty

		if regularization_coefficient:
			val -= regularization_coefficient * uncertainty
		return val.nan_to_num(0).squeeze(), uncertainties.nan_to_num(0).squeeze(1)
	
	@torch.no_grad()
	def _estimate_uncertainty(self, z, action, task):
		q_values = self.model.Q(z, action, task, return_type='all')
		Q_values = math.two_hot_inv(q_values, self.cfg)
		return torch.mean(Q_values, dim=0) * torch.std(Q_values, dim=0)

	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		def compute_loss(actions, z, task):
			actions = actions.unsqueeze(1) 
			z = z.unsqueeze(0)
			value, uncertainties = self._estimate_value(z, actions, task, self.cfg.regularization_coefficient)
			loss = -value

			return loss, (value, uncertainties)
		
		with torch.enable_grad():
			with FreezeParameters([self.model]):
				# Sample policy trajectories
				z = self.model.encode(obs, task)
				if self.cfg.num_pi_trajs > 0:
					pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
					_z = z.repeat(self.cfg.num_pi_trajs, 1)
					for t in range(self.cfg.horizon-1):
						pi_actions[t], _ = self.model.pi(_z, task)
						_z = self.model.next(_z, pi_actions[t], task)
					pi_actions[-1], _ = self.model.pi(_z, task)



				if self.cfg.action_reusage_coefficient and not t0:
					rho = self.cfg.action_reusage_coefficient

					if self.cfg.action_reuse_strategy == "original":
						reused_actions = torch.roll(
							self._prev_planned_actions,
							shifts=-1,
							dims=0,
						)
						reused_actions[-1] = reused_actions[-2]

						pi_actions = (
							rho * reused_actions
							+ (1 - rho) * pi_actions
						)

					elif self.cfg.action_reuse_strategy == "mean":
						mean_prev_actions = self._prev_planned_actions.mean(
							dim=1
						)

						mean_prev_actions = torch.roll(
							mean_prev_actions,
							shifts=-1,
							dims=0,
						)
						mean_prev_actions[-1] = mean_prev_actions[-2]

						reused_actions = mean_prev_actions.unsqueeze(1)

						pi_actions = (
							rho * reused_actions
							+ (1 - rho) * pi_actions
						)

					elif self.cfg.action_reuse_strategy == "best_j":
						best_prev_actions = torch.roll(
							self._prev_best_actions,
							shifts=-1,
							dims=0,
						)
						best_prev_actions[-1] = best_prev_actions[-2]

						reused_actions = best_prev_actions.unsqueeze(1)

						pi_actions = (
							rho * reused_actions
							+ (1 - rho) * pi_actions
						)

					elif self.cfg.action_reuse_strategy == "topk":
						elite_actions = torch.roll(
							self._prev_elite_actions,
							shifts=-1,
							dims=0,
						)
						elite_actions[-1] = elite_actions[-2]

						k = min(
							self.cfg.elite_k,
							self.cfg.num_pi_trajs,
						)

						pi_actions[:, :k] = (
							rho * elite_actions[:, :k]
							+ (1 - rho) * pi_actions[:, :k]
						)

					elif self.cfg.action_reuse_strategy == "matching":
						# Shift previous optimized trajectories by one MPC step.
						shifted_prev_actions = torch.roll(
							self._prev_planned_actions,
							shifts=-1,
							dims=0,
						)
						shifted_prev_actions[-1] = shifted_prev_actions[-2]

						# ---------------------------------------------------------
						# Build pairwise trajectory cost matrix.
						#
						# shifted_prev_actions: [H, N, A]
						# pi_actions:           [H, N, A]
						#
						# Compare only the genuinely overlapping part of horizon.
						# ---------------------------------------------------------
						if self.cfg.horizon > 1:
							old_for_matching = shifted_prev_actions[:-1]
							new_for_matching = pi_actions[:-1]
						else:
							old_for_matching = shifted_prev_actions
							new_for_matching = pi_actions

						# [H, N, A] -> [N, H, A]
						old_trajs = old_for_matching.permute(1, 0, 2)
						new_trajs = new_for_matching.permute(1, 0, 2)

						# Pairwise differences:
						#
						# [N, 1, H, A]
						# -
						# [1, N, H, A]
						#
						# -> [N_old, N_new, H, A]
						diff = (
							old_trajs.unsqueeze(1)
							- new_trajs.unsqueeze(0)
						)

						# [N_old, N_new]
						cost_matrix = diff.pow(2).mean(dim=(2, 3))

						# Hungarian algorithm:
						# Find the minimum-cost one-to-one assignment.
						row_idx, col_idx = linear_sum_assignment(
							cost_matrix.detach().cpu().numpy()
						)

						row_idx = torch.as_tensor(
							row_idx,
							device=self.device,
							dtype=torch.long,
						)

						col_idx = torch.as_tensor(
							col_idx,
							device=self.device,
							dtype=torch.long,
						)

						# matched_prev[:, j] should contain the old trajectory
						# assigned to current candidate j.
						matched_prev = torch.empty_like(
							shifted_prev_actions
						)

						matched_prev[:, col_idx, :] = (
							shifted_prev_actions[:, row_idx, :]
						)

						# Same reuse equation as original Dream-MPC.
						pi_actions = (
							rho * matched_prev
							+ (1 - rho) * pi_actions
						)

					elif self.cfg.action_reuse_strategy == "topk_matching":
						# ---------------------------------------------------------
						# Top-K Matching:
						# 1. reuse only the K highest-J trajectories from last step
						# 2. match them to K distinct current policy proposals
						# 3. leave the remaining N-K proposals completely fresh
						# ---------------------------------------------------------

						k = min(
							self.cfg.elite_k,
							self.cfg.num_pi_trajs,
						)

						assert self._prev_elite_actions.shape[1] >= k
						assert k <= self.cfg.num_pi_trajs
						# [H, K, A]
						elite_actions = torch.roll(
							self._prev_elite_actions[:, :k],
							shifts=-1,
							dims=0,
						)
						elite_actions[-1] = elite_actions[-2]

						# Match only on the genuinely overlapping horizon.
						if self.cfg.horizon > 1:
							old_for_matching = elite_actions[:-1]
							new_for_matching = pi_actions[:-1]
						else:
							old_for_matching = elite_actions
							new_for_matching = pi_actions

						# old: [H, K, A] -> [K, H, A]
						# new: [H, N, A] -> [N, H, A]
						old_trajs = old_for_matching.permute(1, 0, 2)
						new_trajs = new_for_matching.permute(1, 0, 2)

						# [K, 1, H, A] - [1, N, H, A]
						# -> [K, N, H, A]
						diff = (
							old_trajs.unsqueeze(1)
							- new_trajs.unsqueeze(0)
						)

						# [K, N]
						cost_matrix = diff.pow(2).mean(dim=(2, 3))
						assert cost_matrix.shape == (
							k,
							self.cfg.num_pi_trajs,
						)

						# Select K distinct current proposals for the K elites.
						row_idx, col_idx = linear_sum_assignment(
							cost_matrix.detach().cpu().numpy()
						)

						row_idx = torch.as_tensor(
							row_idx,
							device=self.device,
							dtype=torch.long,
						)

						col_idx = torch.as_tensor(
							col_idx,
							device=self.device,
							dtype=torch.long,
						)

						# Only matched candidates receive historical warm-starts.
						# All other N-K candidates remain pure policy proposals.
						pi_actions[:, col_idx, :] = (
							rho * elite_actions[:, row_idx, :]
							+ (1 - rho) * pi_actions[:, col_idx, :]
						)

					elif self.cfg.action_reuse_strategy == "topk_latent_matching":
						# ---------------------------------------------------------
						# Top-K Latent Matching:
						# 1. keep the K highest-J trajectories from the last MPC step
						# 2. shift them forward by one step
						# 3. roll both historical elites and current policy proposals
						#    forward from the current latent state
						# 4. match them using latent-trajectory distance
						# 5. warm-start only the matched K current candidates
						# ---------------------------------------------------------

						k = min(
							self.cfg.elite_k,
							self.cfg.num_pi_trajs,
						)

						assert self._prev_elite_actions.shape[1] >= k
						assert k <= self.cfg.num_pi_trajs

						# ---------------------------------------------------------
						# Shift previous elite action trajectories.
						#
						# elite_actions: [H, K, A]
						# ---------------------------------------------------------
						elite_actions = torch.roll(
							self._prev_elite_actions[:, :k],
							shifts=-1,
							dims=0,
						)

						elite_actions[-1] = elite_actions[-2]

						# ---------------------------------------------------------
						# Ignore the padded last historical action when matching.
						#
						# For H=3:
						# match_horizon = 2
						# ---------------------------------------------------------
						if self.cfg.horizon > 1:
							old_match_actions = elite_actions[:-1]
							new_match_actions = pi_actions[:-1]
						else:
							old_match_actions = elite_actions
							new_match_actions = pi_actions

						match_horizon = old_match_actions.shape[0]

						# ---------------------------------------------------------
						# Roll historical elites and current proposals forward
						# from exactly the same current latent state z.
						#
						# old_z: [K, D]
						# new_z: [N, D]
						# ---------------------------------------------------------
						with torch.no_grad():
							old_z = z.repeat(k, 1)

							new_z = z.repeat(
								self.cfg.num_pi_trajs,
								1,
							)

							old_latents = []
							new_latents = []

							for t in range(match_horizon):
								old_z = self.model.next(
									old_z,
									old_match_actions[t],
									task,
								)

								new_z = self.model.next(
									new_z,
									new_match_actions[t],
									task,
								)

								old_latents.append(old_z)
								new_latents.append(new_z)

							# [T, K, D]
							old_latents = torch.stack(
								old_latents,
								dim=0,
							)

							# [T, N, D]
							new_latents = torch.stack(
								new_latents,
								dim=0,
							)

						# ---------------------------------------------------------
						# [T, K, D] -> [K, T, D]
						# [T, N, D] -> [N, T, D]
						# ---------------------------------------------------------
						old_latents = old_latents.permute(1, 0, 2)
						new_latents = new_latents.permute(1, 0, 2)

						# ---------------------------------------------------------
						# Pairwise latent trajectory distance.
						#
						# [K, 1, T, D]
						# -
						# [1, N, T, D]
						#
						# -> [K, N, T, D]
						# ---------------------------------------------------------
						latent_diff = (
							old_latents.unsqueeze(1)
							- new_latents.unsqueeze(0)
						)

						# [K, N]
						cost_matrix = latent_diff.pow(2).mean(
							dim=(2, 3)
						)

						assert cost_matrix.shape == (
							k,
							self.cfg.num_pi_trajs,
						)

						assert torch.isfinite(cost_matrix).all()

						# ---------------------------------------------------------
						# Hungarian matching:
						#
						# row_idx -> historical elite index
						# col_idx -> current proposal index
						# ---------------------------------------------------------
						row_idx, col_idx = linear_sum_assignment(
							cost_matrix.detach().cpu().numpy()
						)

						row_idx = torch.as_tensor(
							row_idx,
							device=self.device,
							dtype=torch.long,
						)

						col_idx = torch.as_tensor(
							col_idx,
							device=self.device,
							dtype=torch.long,
						)

						# ---------------------------------------------------------
						# Only the K matched candidates receive historical reuse.
						# Remaining N-K candidates stay completely fresh.
						# ---------------------------------------------------------
						pi_actions[:, col_idx, :] = (
							rho * elite_actions[:, row_idx, :]
							+ (1 - rho) * pi_actions[:, col_idx, :]
						)

					elif self.cfg.action_reuse_strategy == "topk_objective_matching":
						# ---------------------------------------------------------
						# Top-K Objective-Aware Matching:
						#
						# 1. keep the K highest-J trajectories from last MPC step
						# 2. shift them forward by one step
						# 3. construct every elite-policy mixed trajectory
						# 4. evaluate all K x N mixed trajectories using
						#    Dream-MPC's own planner objective J
						# 5. find the assignment maximizing total J
						# 6. warm-start only those K matched candidates
						#
						# Remaining N-K candidates stay completely fresh.
						# ---------------------------------------------------------

						k = min(
							self.cfg.elite_k,
							self.cfg.num_pi_trajs,
						)

						n = self.cfg.num_pi_trajs

						assert self._prev_elite_actions.shape[1] >= k
						assert k <= n

						# ---------------------------------------------------------
						# Shift previous elite trajectories by one MPC step.
						#
						# [H, K, A]
						# ---------------------------------------------------------
						elite_actions = torch.roll(
							self._prev_elite_actions[:, :k],
							shifts=-1,
							dims=0,
						)

						elite_actions[-1] = elite_actions[-2]

						# ---------------------------------------------------------
						# Build all K x N possible mixed trajectories.
						#
						# elite_actions.unsqueeze(2): [H, K, 1, A]
						# pi_actions.unsqueeze(1):    [H, 1, N, A]
						#
						# mixed_actions:              [H, K, N, A]
						# ---------------------------------------------------------
						mixed_actions = (
							rho * elite_actions.unsqueeze(2)
							+ (1 - rho) * pi_actions.unsqueeze(1)
						)

						assert mixed_actions.shape == (
							self.cfg.horizon,
							k,
							n,
							self.cfg.action_dim,
						)

						# ---------------------------------------------------------
						# Flatten K x N combinations so that compute_loss can
						# evaluate them in one vectorized call.
						#
						# [H, K, N, A]
						# ->
						# [H, K*N, A]
						# ---------------------------------------------------------
						mixed_actions_flat = mixed_actions.reshape(
							self.cfg.horizon,
							k * n,
							self.cfg.action_dim,
						)

						# Every candidate starts from the SAME current latent state.
						#
						# [K*N, D]
						objective_z = z.repeat(k * n, 1)

						# Mirror the task handling used by the normal planner.
						if task is not None:
							objective_tasks = task.repeat(k * n, 1)
						else:
							objective_tasks = torch.empty(
								k * n,
								device=self.device,
							)

						# ---------------------------------------------------------
						# Evaluate every mixed trajectory using Dream-MPC's
						# planner objective.
						#
						# compute_loss returns:
						#
						# loss = -value
						# aux  = (value, uncertainties)
						#
						# We want to MAXIMIZE value.
						# ---------------------------------------------------------
						with torch.no_grad():
							_, (mixed_returns, _) = vmap(
								compute_loss,
								in_dims=(1, 0, 0),
								randomness="same",
							)(
								mixed_actions_flat,
								objective_z,
								objective_tasks,
							)

						# ---------------------------------------------------------
						# Restore the K x N structure.
						#
						# score_matrix[i, j]
						#
						# = planner objective J of:
						#
						# rho * elite_i
						# +
						# (1-rho) * proposal_j
						#
						# [K, N]
						# ---------------------------------------------------------
						score_matrix = mixed_returns.reshape(
							k,
							n,
						)

						assert score_matrix.shape == (k, n)
						assert torch.isfinite(score_matrix).all()

						# ---------------------------------------------------------
						# Hungarian solves a MINIMIZATION problem.
						#
						# We want:
						#
						# maximize sum J
						#
						# which is equivalent to:
						#
						# minimize sum (-J)
						# ---------------------------------------------------------
						row_idx, col_idx = linear_sum_assignment(
							(-score_matrix).detach().cpu().numpy()
						)

						row_idx = torch.as_tensor(
							row_idx,
							device=self.device,
							dtype=torch.long,
						)

						col_idx = torch.as_tensor(
							col_idx,
							device=self.device,
							dtype=torch.long,
						)

						# ---------------------------------------------------------
						# Apply historical reuse ONLY to the K selected proposals.
						#
						# row_idx -> elite index
						# col_idx -> current proposal index
						#
						# Remaining N-K proposals stay unchanged / fresh.
						# ---------------------------------------------------------
						pi_actions[:, col_idx, :] = (
							rho * elite_actions[:, row_idx, :]
							+ (1 - rho) * pi_actions[:, col_idx, :]
						)


					else:
						raise ValueError(
							f"Unknown action reuse strategy: "
							f"{self.cfg.action_reuse_strategy}"
						)

				# Initialize state and parameters
				z = z.repeat(self.cfg.num_pi_trajs, 1)
				if task:
					tasks = task.repeat(self.cfg.num_pi_trajs, 1)
				else:
					tasks = torch.empty(self.cfg.num_pi_trajs, device=self.device)

				if self.cfg.multitask:
					pi_actions = pi_actions * self.model._action_masks[task]

				actions = nn.Parameter(pi_actions)
				optimizer = torch.optim.SGD([actions], lr=self.cfg.mpc_lr)

				# Iterate gradient-based MPC
				for _ in range(self.cfg.iterations):
					optimizer.zero_grad()

					ft_per_candidate_grads, (returns, uncertainties) = vmap(grad(compute_loss, has_aux=True), in_dims=(1, 0, 0), randomness="same")(
						actions, z, tasks,
					)
					actions.grad = ft_per_candidate_grads.permute(1, 0, 2).clone().detach()
					torch.nn.utils.clip_grad_norm_(actions, self.cfg.grad_clip_norm)

					optimizer.step()

					if self.cfg.multitask:
						actions = actions * self.model._action_masks[task]

				
				# Preserve the original Dream-MPC action-selection behavior.
				idx = returns.detach().argmax(dim=0)
				best_actions = actions[:, idx].detach().clamp(-1, 1)

				final_actions = actions.detach().clamp(-1, 1)

				# Used by Original / Mean.
				self._prev_planned_actions = final_actions

				# Best-J and Top-K both need the final planner scores.
				if self.cfg.action_reuse_strategy in (
					"best_j",
					"topk",
					"topk_matching",
					"topk_latent_matching",
					"topk_objective_matching",
				):
					with torch.no_grad():
						_, (final_returns, _) = vmap(
							compute_loss,
							in_dims=(1, 0, 0),
							randomness="same",
						)(
							final_actions,
							z,
							tasks,
						)

					if self.cfg.action_reuse_strategy == "best_j":
						final_best_idx = final_returns.detach().argmax(dim=0)

						self._prev_best_actions = (
							final_actions[:, final_best_idx]
						)

					elif self.cfg.action_reuse_strategy in (
						"topk",
						"topk_matching",
						"topk_latent_matching",
						"topk_objective_matching",
					):
						k = min(
							self.cfg.elite_k,
							self.cfg.num_pi_trajs,
						)

						topk_idx = torch.topk(
							final_returns.detach(),
							k=k,
							dim=0,
							largest=True,
							sorted=True,
						).indices

						self._prev_elite_actions = (
							final_actions[:, topk_idx]
						)

				return best_actions[0]

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
			"pi_log_std": info["log_std"],
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer, pretrain=False):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task, info = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, **kwargs)
