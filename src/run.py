import yaml
import time
import torch
import datetime
import numpy as np
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from src.agents import RandomAgent
from src.utils import (
    device,
    set_seed,
    make_environment,
)


DEVICE = device()
HYPER_PARAMS_PATH: str='configs/hyper_params.yaml'


def tensor(x: np.array, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=type, device=device)


def zeros(x: tuple, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=type, device=device)


def run(hparams):
    start_time = time.time()
    
    # load hyper-params if not provided
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)
    
    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)
    
    # save parameters and/or configs if you wish
    run_name = (
        f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        'hyperparameters',
        "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            )
        ),
    )
    
    set_seed(args.seed)

    env = make_environment(args.environment)
    agent = RandomAgent(environment=env)

    # start rolling
    observation, info = env.reset()
    for step in range(args.total_timesteps):
        
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(
            action
        )

        # just a dummy log to give you an example
        writer.add_scalar(
            'charts/steps', step, step
        )

        if terminated or truncated:
            
            et = time.time()-start_time
            et = str(datetime.timedelta(seconds=round(et)))

            logger.info(
                f'step={step}, ' + \
                f'episodic_return={info["episode"]["r"]}, ' + \
                f'episodic_length={info["episode"]["l"]}, ' + \
                f'time_elapsed={et}'
            )
            
            writer.add_scalar(
                'charts/episodic_return', info['episode']['r'], step
            )
            writer.add_scalar(
                'charts/episodic_length', info['episode']['l'], step
            )

            break
    
    # save your agent/model often
    torch.save(
        agent, f'{agent.name}.torch'
    )
    logger.info(
        f'Model {agent.name} saved.'
    )

    env.close()
    writer.close()
