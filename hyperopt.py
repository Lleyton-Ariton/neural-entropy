import os

import json

import logging
import argparse

import numpy as np

import functools

import optuna

import lsm

from objectives import (
    construct_transition_matrix,
    topological_entropy_objective,
    shannon_entropy_objective,
    entropy_rate_objective
)

from typing import Union, Optional, List, Dict


def objective(
        trial: optuna.Trial,
        sweep: Union[
            str, Dict[str, Union[Dict[str, List[int]], List[int]]]
        ],
        steps: int,
        optimization_objective: str = 'topological',
        trials: int = 1,
        device: str = 'cpu',
        dtype: str = 'f32',
        dt: float = 0.1,
        logger: Optional[logging.Logger] = None
) -> float:
    if optimization_objective not in ['topological', 'shannon', 'rate']:
        raise ValueError(
            f'Expected the optimization objective to be either '
            f'of \'topological\', \'shannon\' or \'rate\', '
            f'but received {optimization_objective} instead.'
        )

    if isinstance(sweep, str):
        if os.path.exists(sweep):
            with open(sweep, 'r') as file:
                params = json.load(file)

                if not isinstance(params, dict):
                    raise ValueError(
                        'An error occurred when loading'
                        ' the sweep parameter file.'
                    )
        else:
            raise ValueError(
                f'The provided \'sweep_parameters\' must represent a valid '
                f'path if given as a string. No \'{sweep}\' exists.'
            )

    metric_value = 0.0
    for i in range(trials):
        network = lsm.LiquidStateMachine(
            **{
                'inputs': params['network']['inputs'][0],
                'size': params['network']['size'][0],
                'lambda': trial.suggest_categorical('lambda', params['lambda']),
                'constant': trial.suggest_categorical('constant',
                                                      params['constant']),
                'excitatory_ratio': trial.suggest_categorical(
                    'excitatory_ratio', params['excitatory_ratio']
                ),
                'input_sparsity': trial.suggest_categorical(
                    'input_sparsity', params['input_sparsity']
                ),
                'input_synapses': lsm.Fixed(),
                'input_weight_distribution':
                    lsm.UniformWeightDistribution(
                        trial.suggest_categorical(
                            'input_weight_distribution_low',
                            params['inputs']['low']
                        ),
                        trial.suggest_categorical(
                            'input_weight_distribution_high',
                            params['inputs']['high']
                        ),
                    ),
                'excitatory_synapses': lsm.Fixed(),
                'excitatory_weight_distribution':
                    lsm.LogNormalWeightDistribution(
                        trial.suggest_categorical(
                            'excitatory_weight_distribution_mean',
                            params['excitatory']['means']
                        ),
                        trial.suggest_categorical(
                            'excitatory_weight_distribution_deviation',
                            params['excitatory']['deviations']
                        ),
                    ),
                'inhibitory_weight_distribution':
                    lsm.LogNormalWeightDistribution(
                        trial.suggest_categorical(
                            'inhibitory_weight_distribution_mean',
                            params['inhibitory']['means']
                        ),
                        trial.suggest_categorical(
                            'inhibitory_weight_distribution_deviation',
                            params['inhibitory']['deviations']
                        ),
                    )
            },
            device=device,
            dtype=dtype
        )

        states = np.zeros((steps, network.size()))

        results = network.run(steps, dt=dt)
        for (time, index) in results:
            states[time, index] = 1.0

        if optimization_objective == 'topological':
            transition_matrix, _, _ = construct_transition_matrix(states)

            metric_value += topological_entropy_objective(transition_matrix)
        elif optimization_objective == 'shannon':
            transition_matrix, _, _ = construct_transition_matrix(states)

            metric_value += shannon_entropy_objective(transition_matrix)
        else:
            transition_matrix, _, _ = construct_transition_matrix(states)

            metric_value += entropy_rate_objective(transition_matrix)

        trial.report(metric_value / (i + 1), i)

        if trial.should_prune():
            raise optuna.TrialPruned()

    metric_value = metric_value / trials

    if logger:
        logger.info(
            f'{optimization_objective.title()} Entropy: {metric_value: .4f}'
        )

    return metric_value


def main(config: argparse.Namespace, logger: logging.Logger) -> None:
    if config.device.startswith('cuda'):
        device = config.device
    else:
        device = 'cpu'

    logging.info(f'Using the device \'{device}\'...')
    logger.info(
        f'Optimization objective is set to \'{config.optimization_objective}\'.'
    )

    objective_wrapper = functools.partial(
        objective,
        sweep=config.sweep_file,
        steps=config.steps,
        optimization_objective=config.optimization_objective,
        trials=config.simulation_trials,
        device=device,
        dtype=config.dtype,
        dt=config.dt,
        logger=logger
    )

    logger.info('Starting optimization...')

    study = optuna.create_study(
        study_name=config.study_name,
        storage=optuna.storages.RDBStorage(
            'postgresql://endpoint/postgres'  # Removed for privacy.
        ),
        direction='maximize',
        load_if_exists=True
    )

    study.optimize(objective_wrapper, n_trials=config.n_trials)

    logging.info('Completed optimization trials.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sweep_file',
        type=str,
        required=True,
        help='Path to the parameter sweep configuration file. '
             'The file must be of type \'.json\' and contain keys '
             'in direct correspondence with the paramaters to be swept.'
    )
    parser.add_argument(
        '--optimization_objective',
        type=str,
        default='topological',
        help='The specific objective to optimize. '
             'Can be either of \'topological\', \'shannon\' or \'rate\'.'
    )
    parser.add_argument(
        '--simulation_trials',
        type=int,
        default=10,
        help='The number of simulation trials.'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100000,
        help='The number of simulation steps.'
    )
    parser.add_argument(
        '--study_name',
        type=str,
        required=True,
        help='The name of the Optuna hyperparameter optimization study.'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='The number of Optuna trials to perform. '
             'This parameter will default to 100.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Specifies the device on which to run. If attempting to run '
             'trials across multiple GPUs, explicitly set the cuda rank.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='f32',
        help='The data type of the simulation.'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        help='The integration time step.'
    )

    main(config=parser.parse_args(), logger=logging.getLogger(__name__))
