import torch
from torch import nn
import torch.nn.functional as F
from torch.func import grad, vmap

from common import math
from common.planner import FreezeParameters
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict, merge_tensordicts


class DreamMPC_BMPC(torch.nn.Module):
	"""Dream-MPC (BMPC) agent. Implements training + inference.
	Can be used for only single-task experiments at the moment,
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
		self._prev_mean = torch.nn.Buffer(torch.zeros(1, self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self._prev_planned_actions = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
		self.update_count = 0

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
			TensorDict: Planned info.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			action, plan_info = self.plan(obs, batch_size=1, t0=t0, eval_mode=eval_mode, task=task, horizon=self.cfg.horizon)
			return action[0].cpu(), plan_info.cpu()
		z = self.model.encode(obs, task)
		action, pi_info = self.model.pi(z, task)
		if eval_mode:
			action = pi_info["mean"]
		return action[0].cpu(), None

	def _estimate_value(self, z, actions, task, horizon, regularization_coefficient=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		uncertainties = torch.empty_like(actions, device=self.device)
		for t in range(horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
			z = self.model.next(z, actions[:, t], task)
			G = G + discount * reward
			uncertainty = self._estimate_uncertainty(z, actions[:, t], task)
			uncertainties[:, t, :] = uncertainty
			if regularization_coefficient:
				G -= regularization_coefficient * uncertainty
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		if self.cfg.use_v_instead_q:
			action = None  # dummy value to keep the interface
			val = G + discount * self.model.V(z, task, return_type="avg")
		else:
			action, _ = self.model.pi(z, task)
			val = G + discount * self.model.Q(z, action, task, return_type='avg')

		uncertainty = self._estimate_uncertainty(z, action, task)
		uncertainties[:, -1, :] = uncertainty

		if regularization_coefficient:
			val -= regularization_coefficient * uncertainty
		return val.nan_to_num(0).squeeze(), uncertainties.nan_to_num(0)
	
	@torch.no_grad()
	def _estimate_uncertainty(self, z, action, task):
		if self.cfg.use_v_instead_q:
			v_values = self.model.V(z, task, return_type="all")
			V_values = math.two_hot_inv(v_values, self.cfg)
			return torch.mean(V_values, dim=0) * torch.std(V_values, dim=0)
		else:
			q_values = self.model.Q(z, action, task, return_type='all')
			Q_values = math.two_hot_inv(q_values, self.cfg)
			return torch.mean(Q_values, dim=0) * torch.std(Q_values, dim=0)

	def _plan(self, obs, batch_size, t0=False, eval_mode=False, task=None, horizon=None, update_prev_mean=True, reanalyze=False):
		"""
		Plan a batched sequence of actions using the learned world model.

		Batched-plan implementation is borrowed from vectorized_env branch of tdmpc2 repo.
		Perhaps the batch size dimension can be changed to optimize memory access performance.
		
		Args:
			obs (torch.Tensor): observation from which to plan.
			batch_size (int): observation's batch size.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).
			horizon (int): Planning horizon.
			update_prev_mean (bool): Whether to update self._prev_mean using mean,
				when batched plan in reanalyze, update_prev_mean should be False.
			reanalyze (bool): Reanalyzing or not.
		Returns:
			torch.Tensor: Action to take in the environment.
		"""

		def compute_loss(actions, z, task):
			value, uncertainties = self._estimate_value(z, actions, task, horizon, self.cfg.regularization_coefficient)
			loss = -value

			return loss, (value, uncertainties)
		
		with torch.enable_grad():
			with FreezeParameters([self.model]):
				# Sample policy trajectories
				z = self.model.encode(obs, task)
				if self.cfg.num_pi_trajs > 0:
					pi_actions = torch.empty(batch_size, horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
					_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
					for t in range(horizon-1):
						pi_actions[:,t], _ = self.model.pi(_z, task, expl=reanalyze)
						_z = self.model.next(_z, pi_actions[:,t], task)
					pi_actions[:,-1], _ = self.model.pi(_z, task, expl=reanalyze)

				if self.cfg.action_reusage_coefficient and not t0 and not reanalyze:
                    # shift actions by one time step and use the same action as the last for the one beyond the previous prediction horizon
					corresponding_previously_planned_actions = torch.roll(self._prev_planned_actions, shifts=-1, dims=0)
					corresponding_previously_planned_actions[-1] = corresponding_previously_planned_actions[-2]
					pi_actions = self.cfg.action_reusage_coefficient * corresponding_previously_planned_actions.unsqueeze(0).repeat(batch_size, 1, 1, 1) +  (1 - self.cfg.action_reusage_coefficient) * pi_actions

				# Initialize state and parameters
				z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
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

					ft_per_candidate_grads, (returns, uncertainties) = vmap(grad(compute_loss, has_aux=True), in_dims=(2, 1, 0), randomness="same")(
						actions, z, tasks,
					)
					actions.grad = ft_per_candidate_grads.permute(1, 2, 0, 3).clone().detach()
					torch.nn.utils.clip_grad_norm_(actions, self.cfg.grad_clip_norm)

					optimizer.step()

					if self.cfg.multitask:
						actions = actions * self.model._action_masks[task]

				idx = returns.detach().argmax(dim=0)
				best_actions = actions[:, :, idx].detach().clamp(-1, 1)

				if not reanalyze:
					self._prev_planned_actions = actions.squeeze(0).detach().clamp(-1, 1)

				std, mean = torch.std_mean(actions[:, 0].detach(), dim=1)
				std = std.clamp(self.cfg.min_std, self.cfg.max_std)
				action_dist = torch.cat([mean, std], dim=-1)

				info = TensorDict({"action_value": returns.detach()[idx], "action_dist": action_dist})

				return best_actions[:, 0, :], info

	def update_pi(self, zs, expert_action_dist, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		_, info = self.model.pi(zs[:-1], task)
		actions_dist = torch.cat([info["mean"], info["log_std"].exp()], dim=-1)
		kl_loss = math.kl_div(actions_dist, expert_action_dist).mean(-1, keepdim=True)
		self.scale.update(kl_loss[0])
		kl_loss = self.scale(kl_loss)
  
		rho = torch.pow(self.cfg.rho, torch.arange(len(kl_loss), device=self.device))
		pi_loss = ((kl_loss - self.cfg.entropy_coef * info["scaled_entropy"]).mean(dim=(1,2)) * rho).mean()
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
	def _td_target_Q(self, next_z, reward, task):
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
		return reward + discount * self.model.Q(next_z, action, task, return_type='avg', target=True)

	@torch.no_grad()
	def _td_target_V(self, zs, task):
		"""
		Compute the TD-target using learned model.

		Args:
			zs (torch.Tensor): Latent state from observation.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""

		Gs, discount = 0, 1
		zs_ = zs.clone()
		for _ in range(self.cfg.td_horizon):
			actions, _ = self.model.pi(zs_, task)
			rewards = math.two_hot_inv(self.model.reward(zs_, actions, task), self.cfg)
			zs_ = self.model.next(zs_, actions, task)
			Gs += discount * rewards
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		td_target = Gs + discount * self.model.V(zs_, task, return_type="avg", target=True)
		return td_target
	
	def _update(self, obs, action, reward, expert_action_dist, task=None, pretrain=False):
		# Compute targets
		with torch.no_grad():
			true_zs = self.model.encode(obs, task) # latent from real obs
			next_z = true_zs[1:]
			if self.cfg.use_v_instead_q:
				td_targets = self._td_target_V(true_zs[:-1], task)
			else:
				td_targets = self._td_target_Q(next_z, reward, task)
	
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
		if self.cfg.use_v_instead_q:
			qs = self.model.V(_zs, task, return_type='all')
		else:
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

		# Collect training statistics
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
  
		# Update policy
		if not pretrain:
			pi_info = self.update_pi(zs.detach(), expert_action_dist, task)
			info.update(pi_info)

		# Update target Q-functions
		self.model.soft_update_target_Q()
		self.model.eval()

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
		expert_action_dist = info["expert_action_dist"]

		# preprocess expert data (replace nan with Normal(0,1))
		# If using the expert value, should also apply preprocessing here.
		mean, std = expert_action_dist.chunk(2, dim=-1)
		nan_idx = torch.isnan(mean) # samples which are not generated by mpc policy
		mean[nan_idx] = torch.zeros_like(action[nan_idx])
		std[nan_idx] = torch.ones_like(action[nan_idx])
		expert_action_dist = torch.cat([mean, std],dim=-1)
  
		# lazy reanalyze
		if not pretrain:
			self.update_count += 1
			if (self.cfg.reanalyze_interval > 0) and (self.update_count % self.cfg.reanalyze_interval == 0):
				reanalyzed_action_dist = self.reanalyze(buffer, obs, task, info["index"])
				# update reanalyzed sample
				expert_action_dist[:,:self.cfg.reanalyze_batch_size] = reanalyzed_action_dist

		kwargs = {"pretrain": pretrain}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, expert_action_dist, **kwargs)

	@torch.no_grad()
	def reanalyze(self, buffer, obs, task, index):
		'''
		Do lazy reanalyze
		'''
		self.model.eval()
  
		# re-plan
		obs_ = obs[:-1,:self.cfg.reanalyze_batch_size].reshape(self.cfg.horizon*self.cfg.reanalyze_batch_size, *obs.shape[2:])
		torch.compiler.cudagraph_mark_step_begin()

		def _fn(o):
			_, p = self.plan(o.unsqueeze(0), 1, t0=True, task=task, horizon=self.cfg.reanalyze_horizon, update_prev_mean=False, reanalyze=True)
			return p
		
		# vmap doesn't allow non-scalars (batch size) and in-place arithmetic, so we use a for loop for now
		res = []
		for o in obs_:
			p = _fn(o)
			res.append(p)

		plan_info = merge_tensordicts(*res, callback_exist=lambda *v: torch.stack(v))

		# Update reanalyzed data to buffer
		index_list = index[1:, :self.cfg.reanalyze_batch_size].flatten().tolist()
		with buffer._buffer._replay_lock: # Add lock to prevent any unexpected data change due to thread risk
			buffer._buffer._storage._storage['expert_action_dist'][index_list] = \
				plan_info["action_dist"].squeeze(dim=1).to(buffer._buffer._storage.device)
			buffer._buffer._storage._storage['expert_value'][index_list] = \
				plan_info["action_value"].flatten().to(buffer._buffer._storage.device)
    
		self.model.train()
		return plan_info["action_dist"].squeeze(dim=1).view(self.cfg.horizon, self.cfg.reanalyze_batch_size, -1)
