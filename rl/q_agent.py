from typing import Optional
from dataclasses import dataclass, field
import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from networks.encoder import VitEncoder, VitEncoderConfig
from networks.encoder import ResNetEncoder, ResNetEncoderConfig, DrQEncoder
from networks.encoder import ResNet96Encoder, ResNet96EncoderConfig
from rl.critic import Critic, CriticConfig
from rl.actor import Actor, ActorConfig
from rl.actor import FcActor, FcActorConfig
from rl.critic import MultiFcQ, MultiFcQConfig


@dataclass
class QAgentConfig:
    device: str = "cuda"
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_clip: float = 0.3
    # encoder
    use_prop: int = 0
    enc_type: str = "vit"
    vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())
    resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
    resnet96: ResNet96EncoderConfig = field(default_factory=lambda: ResNet96EncoderConfig())
    # critic & actor
    critic: CriticConfig = field(default_factory=lambda: CriticConfig())
    actor: ActorConfig = field(default_factory=lambda: ActorConfig())
    state_critic: MultiFcQConfig = field(default_factory=lambda: MultiFcQConfig())
    state_actor: FcActorConfig = field(default_factory=lambda: FcActorConfig())
    # algo
    act_method: str = "rl"  # "rl/ibrl/ibrl_soft"
    bootstrap_method: str = ""  # "rl/ibrl/ibrl_soft"
    # ibrl
    ibrl_eps_greedy: float = 1
    soft_ibrl_beta: float = 10
    # bc loss regularization
    bc_loss_coef: float = 0.1
    bc_loss_dynamic: int = 0  # dynamically scale bc loss weight

    def __post_init__(self):
        if self.bootstrap_method == "":
            self.bootstrap_method = self.act_method


class QAgent(nn.Module):
    def __init__(
        self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig
    ):
        super().__init__()
        self.use_state = use_state
        self.rl_camera = rl_camera
        self.cfg = cfg

        if use_state:
            self.critic = MultiFcQ(obs_shape, action_dim, cfg.state_critic)
            self.actor = FcActor(obs_shape, action_dim, cfg.state_actor)
        else:
            self.encoder = self._build_encoders(obs_shape)
            repr_dim = self.encoder.repr_dim
            patch_repr_dim = self.encoder.patch_repr_dim
            print("encoder output dim: ", repr_dim)
            print("patch output dim: ", patch_repr_dim)

            assert len(prop_shape) == 1
            prop_dim = prop_shape[0] if cfg.use_prop else 0

            # create critics & actor
            self.critic = Critic(
                repr_dim=repr_dim,
                patch_repr_dim=patch_repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                cfg=self.cfg.critic,
            )
            self.actor = Actor(repr_dim, patch_repr_dim, prop_dim, action_dim, cfg.actor)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        if not self.use_state:
            print(common_utils.wrap_ruler(f"encoder weights"))
            print(self.encoder)
            common_utils.count_parameters(self.encoder)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        if not self.use_state:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.lr)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: Optional[common_utils.MultiCounter] = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    def _build_encoders(self, obs_shape):
        if self.cfg.enc_type == "vit":
            return VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet":
            return ResNetEncoder(obs_shape, self.cfg.resnet).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet96":
            return ResNet96Encoder(obs_shape, self.cfg.resnet96).to(self.cfg.device)
        elif self.cfg.enc_type == "drq":
            return DrQEncoder(obs_shape).to(self.cfg.device)
        else:
            assert False, f"Unknown encoder type {self.cfg.enc_type}."

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        if not self.use_state:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method
        return

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        """This function encodes the observation into feature tensor."""
        data = obs[self.rl_camera].float()
        if augment:
            data = self.aug(data)
        return self.encoder.forward(data, flatten=False)

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if self.use_state:
            if obs["state"].dim() == 1:
                should_unsqueeze = True
        else:
            if obs[self.rl_camera].dim() == 3:
                should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(
        self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True
    ) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        unsqueezed = self._maybe_unsqueeze_(obs)

        if not self.use_state:
            assert "feat" not in obs
            obs["feat"] = self._encode(obs, augment=False)

        if self.cfg.act_method == "rl":
            action = self._act_default(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                use_target=False,
            )
        elif self.cfg.act_method == "ibrl":
            action = self._act_ibrl(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                eps_greedy=self.cfg.ibrl_eps_greedy,
                use_target=False,
            )
        elif self.cfg.act_method == "ibrl_soft":
            action = self._act_ibrl_soft(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                use_target=False,
            )
        else:
            assert False, f"unknown act method {self.cfg.act_method}"

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, stddev)

        if eval_mode:
            assert not self.training

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=clip)

        return action

    def _act_ibrl(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        eps_greedy: float,
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            assert not actor.training

        assert len(self.bc_policies) == 1
        bc_policy = self.bc_policies[0]
        bc_action = bc_policy.act(obs, cpu=False)

        rl_dist: utils.TruncatedNormal = actor(obs, stddev)
        if eval_mode:
            rl_action = rl_dist.mean
        else:
            rl_action = rl_dist.sample(clip)

        rl_bc_actions = torch.stack([rl_action, bc_action], dim=1)
        bsize, num_action, _ = rl_bc_actions.size()

        # get q(a)
        # feat -> [batch, num_patch, patch_dim] -> [batch, num_action(2), num_patch, patch_dim]
        flat_actions = rl_bc_actions.flatten(0, 1)
        if isinstance(self.critic_target, Critic):
            flat_qfeats = obs["feat"].unsqueeze(1).repeat(1, num_action, 1, 1).flatten(0, 1)
            flat_props = obs["prop"].unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            q1, q2 = self.critic_target.forward(flat_qfeats, flat_props, flat_actions)
            qa: torch.Tensor = torch.min(q1, q2).view(bsize, num_action)
        else:
            state = obs["state"]
            flat_state = state.unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            qa: torch.Tensor = self.critic_target.forward_k(flat_state, flat_actions)
            qa = qa.min(-1)[0].view(bsize, num_action)

        # best_action_idx: [batch]
        greedy_action_idx: torch.Tensor = qa.argmax(1)
        greedy_action = rl_bc_actions[range(bsize), greedy_action_idx]
        # actions: [batch, action_dim]

        if eval_mode or eps_greedy == 1:
            action = greedy_action
            selected_action_idx = greedy_action_idx
        else:
            eps = torch.rand((bsize, 1), device=qa.device)
            use_greedy = (eps < eps_greedy).float()
            rand_action_idx = torch.randint(0, num_action, (bsize,))
            rand_action = rl_bc_actions[range(bsize), rand_action_idx]
            assert rand_action.size() == greedy_action.size()
            action = rand_action * (1 - use_greedy) + greedy_action * use_greedy
            selected_action_idx = rand_action_idx * (1 - use_greedy) + greedy_action_idx * use_greedy

            if self.stats is not None:
                self.stats["actor/greedy"].append(use_greedy.sum(), bsize)

        if self.stats is not None:
            use_bc = (selected_action_idx >= 1).float()
            if use_target:
                use_bc = use_bc.mean().item()
                self.stats["actor/bootstrap_bc"].append(use_bc)
            else:
                use_bc = use_bc.sum().item()
                if eval_mode:
                    self.stats["actor/bc_eval"].append(use_bc, bsize)
                else:
                    assert bsize == 1, f"bsize should be 1, but got {bsize}"
                    rl_action_norm = rl_action.squeeze()[:6].norm().item()
                    bc_action_norm = bc_action.squeeze()[:6].norm().item()
                    self.stats["actor/anorm_rl"].append(rl_action_norm)
                    self.stats["actor/anorm_bc"].append(bc_action_norm)
                    self.stats["actor/bc_train"].append(use_bc, bsize)

        return action

    def _act_ibrl_soft(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
    ):
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            assert not actor.training

        assert len(self.bc_policies) == 1
        bc_policy = self.bc_policies[0]
        bc_action = bc_policy.act(obs, cpu=False)

        rl_dist: utils.TruncatedNormal = actor(obs, stddev)
        if eval_mode:
            rl_action = rl_dist.mean
        else:
            rl_action = rl_dist.sample(clip)

        rl_bc_actions = torch.stack([rl_action, bc_action], dim=1)

        # cat along the num action dim
        # actions: [bsize, n_rl_actions + n_bc_actions * n_bc, action_dim]
        bsize, num_action, _ = rl_bc_actions.size()

        flat_actions = rl_bc_actions.flatten(0, 1)
        if isinstance(self.critic_target, Critic):
            flat_qfeats = obs["feat"].unsqueeze(1).repeat(1, num_action, 1, 1).flatten(0, 1)
            flat_props = obs["prop"].unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            q1, q2 = self.critic_target.forward(flat_qfeats, flat_props, flat_actions)
            qa: torch.Tensor = torch.min(q1, q2).view(bsize, num_action)
        else:
            state = obs["state"]
            flat_state = state.unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            qa: torch.Tensor = self.critic_target.forward_k(flat_state, flat_actions)
            qa = qa.min(-1)[0].view(bsize, num_action)

        # decide which action to take
        p_center = torch.nn.functional.softmax(qa * self.cfg.soft_ibrl_beta, dim=1)
        center_idx = p_center.multinomial(1)
        if (not use_target) and self.stats is not None and (not eval_mode):
            assert bsize == 1
            self.stats["actor/p_max"].append(p_center.max().item())

        # center_idx: [batchsize, 1]
        action = rl_action * (1 - center_idx) + bc_action * center_idx

        if self.stats is not None:
            use_bc = center_idx.sum().item()
            if use_target:
                # must be called from update_critic
                self.stats["actor/bootstrap_bc"].append(use_bc, bsize)
            else:
                if eval_mode:
                    self.stats["actor/bc_eval"].append(use_bc, bsize)
                else:
                    assert bsize == 1
                    self.stats["actor/bc_act"].append(use_bc, bsize)
                    self.stats["actor/qrl-qbc"].append((qa[0][0] - qa[0][1]).item())

        return action

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
    ):
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            if self.cfg.bootstrap_method == "rl":
                next_action = self._act_default(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    use_target=True,
                )
            elif self.cfg.bootstrap_method == "ibrl":
                next_action = self._act_ibrl(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    eps_greedy=1.0,
                    use_target=True,
                )
            elif self.cfg.bootstrap_method == "ibrl_soft":
                next_action = self._act_ibrl_soft(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    use_target=True,
                )
            else:
                assert False, f"unknown bootstrap method {self.cfg.bootstrap_method}"

            if isinstance(self.critic_target, Critic):
                target_q1, target_q2 = self.critic_target.forward(
                    next_obs["feat"], next_obs["prop"], next_action
                )
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = self.critic_target.forward_k(next_obs["state"], next_action).min(-1)[0]

            target_q = (reward + (discount * target_q)).detach()

        action = reply["action"]
        loss_fn = nn.functional.mse_loss
        if isinstance(self.critic, Critic):
            q1, q2 = self.critic.forward(obs["feat"], obs["prop"], action)
            critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)
        else:
            qs: torch.Tensor = self.critic(obs["state"], action)
            critic_loss = nn.functional.mse_loss(
                qs, target_q.unsqueeze(1).repeat(1, qs.size(1)), reduction="none"
            )
            critic_loss = critic_loss.sum(1).mean(0)

        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()

        if not self.use_state:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        if not self.use_state:
            self.encoder_opt.step()
        self.critic_opt.step()
        return metrics

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        if not self.use_state:
            assert "feat" in obs, "safety check"

        action: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=stddev,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )

        if isinstance(self.critic, Critic):
            q = torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
        else:
            q: torch.Tensor = self.critic(obs["state"], action).min(-1)[0]
        actor_loss = -q.mean()
        return actor_loss

    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        obs: dict[str, torch.Tensor] = batch.obs

        if not self.use_state:
            assert "feat" not in obs, "safety check"
            obs["feat"] = self._encode(obs, augment=True)

        if not backprop_encoder and not self.use_state:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        action: torch.Tensor = batch.action["action"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss

    def update_actor(self, obs: dict[str, torch.Tensor], stddev: float):
        metrics = {}
        actor_loss = self._compute_actor_loss(obs, stddev)
        metrics["train/actor_loss"] = actor_loss.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return metrics

    def update_actor_rft(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: "QAgent",
    ):
        metrics = {}
        actor_loss = self._compute_actor_loss(obs, stddev)
        metrics["train/actor_loss"] = actor_loss.item()
        bc_loss = self._compute_actor_bc_loss(bc_batch, backprop_encoder=False)
        assert actor_loss.size() == bc_loss.size()

        ratio = 1
        if self.cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.cfg.act_method
                self.cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                if isinstance(self.critic, Critic):
                    curr_q = torch.min(*self.critic(bc_obs["feat"], bc_obs["prop"], curr_action))
                    ref_q = torch.min(*self.critic(bc_obs["feat"], bc_obs["prop"], ref_action))
                else:
                    curr_q = self.critic.forward_k(bc_obs["state"], curr_action).min(-1)[0]
                    ref_q = self.critic.forward_k(bc_obs["state"], ref_action).min(-1)[0]

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: Optional["QAgent"] = None,
    ):
        obs: dict[str, torch.Tensor] = batch.obs
        reward: torch.Tensor = batch.reward
        discount: torch.Tensor = batch.bootstrap
        next_obs: dict[str, torch.Tensor] = batch.next_obs

        if not self.use_state:
            obs["feat"] = self._encode(obs, augment=True)
            with torch.no_grad():
                next_obs["feat"] = self._encode(next_obs, augment=True)

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()
        critic_metric = self.update_critic(
            obs=obs,
            reply=batch.action,
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            stddev=stddev,
        )
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        # NOTE: actor loss does not backprop into the encoder
        if not self.use_state:
            obs["feat"] = obs["feat"].detach()

        if bc_batch is None:
            actor_metric = self.update_actor(obs, stddev)
        else:
            assert ref_agent is not None
            actor_metric = self.update_actor_rft(obs, stddev, bc_batch, ref_agent)

        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)

        return metrics

    def pretrain_actor_with_bc(self, batch):
        """pretrain actor and encoder with bc"""
        loss = self._compute_actor_bc_loss(batch, backprop_encoder=True)

        if not self.use_state:
            self.encoder_opt.zero_grad(set_to_none=True)

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()

        if not self.use_state:
            self.encoder_opt.step()
        self.actor_opt.step()

        return {"pretrain/loss": loss.item()}
