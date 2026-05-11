from tdmpc2 import TDMPC2
from bmpc import BMPC
from dream_mpc_bmpc import DreamMPC_BMPC
from dream_mpc_tdmpc2 import DreamMPC_TDMPC2


MODEL_SIZE = { # parameters (M)
	1:   {'enc_dim': 256,
		  'mlp_dim': 384,
		  'latent_dim': 128,
		  'num_enc_layers': 2,
		  'num_q': 2},
	5:   {'enc_dim': 256,
		  'mlp_dim': 512,
		  'latent_dim': 512,
		  'num_enc_layers': 2},
	19:  {'enc_dim': 1024,
		  'mlp_dim': 1024,
		  'latent_dim': 768,
		  'num_enc_layers': 3},
	48:  {'enc_dim': 1792,
		  'mlp_dim': 1792,
		  'latent_dim': 768,
		  'num_enc_layers': 4},
	317: {'enc_dim': 4096,
		  'mlp_dim': 4096,
		  'latent_dim': 1376,
		  'num_enc_layers': 5,
		  'num_q': 8},
}

TASK_SET = {
	'mt30': [
		# 19 original dmcontrol tasks
		'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
	    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
		'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
		'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop',
		# 11 custom dmcontrol tasks
		'walker-walk-backwards', 'walker-run-backwards', 'cheetah-run-backwards', 'cheetah-run-front', 'cheetah-run-back',
		'cheetah-jump', 'hopper-hop-backwards', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
		'pendulum-spin',
	],
	'mt80': [
		# 19 original dmcontrol tasks
		'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
	    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
		'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
		'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop',
		# 11 custom dmcontrol tasks
		'walker-walk-backwards', 'walker-run-backwards', 'cheetah-run-backwards', 'cheetah-run-front', 'cheetah-run-back',
		'cheetah-jump', 'hopper-hop-backwards', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
		'pendulum-spin',
		# meta-world mt50
		'mw-assembly', 'mw-basketball', 'mw-button-press-topdown', 'mw-button-press-topdown-wall', 'mw-button-press',
		'mw-button-press-wall', 'mw-coffee-button', 'mw-coffee-pull', 'mw-coffee-push', 'mw-dial-turn',
		'mw-disassemble', 'mw-door-open', 'mw-door-close', 'mw-drawer-close', 'mw-drawer-open',
		'mw-faucet-open', 'mw-faucet-close', 'mw-hammer', 'mw-handle-press-side', 'mw-handle-press',
		'mw-handle-pull-side', 'mw-handle-pull', 'mw-lever-pull', 'mw-peg-insert-side', 'mw-peg-unplug-side',
		'mw-pick-out-of-hole', 'mw-pick-place', 'mw-pick-place-wall', 'mw-plate-slide', 'mw-plate-slide-side',
		'mw-plate-slide-back', 'mw-plate-slide-back-side', 'mw-push-back', 'mw-push', 'mw-push-wall',
		'mw-reach', 'mw-reach-wall', 'mw-shelf-place', 'mw-soccer', 'mw-stick-push',
		'mw-stick-pull', 'mw-sweep-into', 'mw-sweep', 'mw-window-open', 'mw-window-close',
		'mw-bin-picking', 'mw-box-close', 'mw-door-lock', 'mw-door-unlock', 'mw-hand-insert',
	],
}


def get_agent_cls(cfg):
    if cfg.algo == "bmpc":
        return BMPC
    elif cfg.algo == "tdmpc2":
        return TDMPC2
    elif cfg.algo == "dream_mpc_bmpc":
        return DreamMPC_BMPC
    elif cfg.algo == "dream_mpc_tdmpc2":
        return DreamMPC_TDMPC2
    else:
        raise NotImplementedError
