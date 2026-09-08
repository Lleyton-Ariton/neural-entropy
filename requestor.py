import argparse

import datetime
import logging

import json

import asyncio

import itertools

from yapapi import Golem, Task

from yapapi.payload import vm
from yapapi.log import enable_default_logger

from worker import worker

from typing import (
    Union, Any, Dict, List
)


# Type alias for improving cleanliness and readability.
Sweep = Union[
    Dict[str, Union[List[Any], Dict[str, List[Any]]]],
    List[Dict[str, Union[List[Any], Dict[str, List[Any]]]]]
]


# Helper function for loading and/or sharding a sweep configuration file.
# Remark that any given sweep configuration file must be in JSON format.
def load_sweep_config_file(filename: str, shard: bool = False) -> Sweep:
    with open(filename, 'r') as file:
        sweep = json.load(file)

    return sweep


# Main execution function responsible for requesting all tasks on the client side.
async def main(
        image_hash: str,
        budget: int,
        network: str,
        sweep: Sweep,
        simulation_duration: int = 100000,
        simulation_dt: float = 0.1,
        min_mem_gib: float = 1.0,
        min_cpu_threads: int = 1,
        # timeout: int = 30
):
    combinations = itertools.product(
        sweep['network']['inputs'], sweep['network']['size'],
        sweep['lambda'],
        sweep['constant'],
        sweep['excitatory_ratio'],
        sweep['input_sparsity'],
        sweep['inputs']['low'], sweep['inputs']['high'],
        sweep['excitatory']['means'], sweep['excitatory']['deviations'],
        sweep['inhibitory']['means'], sweep['inhibitory']['deviations']
    )

    tasks = (
        Task(data={
            'parameters': parameters,
            'duration': simulation_duration,
            'dt': simulation_dt
        }) for parameters in combinations
    )

    logger.info(f'Created tasks.')
    logger.info('Starting parameter sweep execution...')
    async with Golem(
            budget=budget,
            payment_network=network,
            subnet_tag='public'
    ) as golem:
        try:
            payload = await vm.repo(
                image_hash=image_hash,
                min_mem_gib=min_mem_gib,
                min_cpu_threads=min_cpu_threads
            )

            results = golem.execute_tasks(
                worker,
                tasks,
                payload=payload,
                # max_workers=None,
                # timeout=datetime.timedelta(minutes=timeout)
            )

            completed, failed = [], []
            async for task in results:
                if task.is_failed or task.is_rejected:
                    failed.append(task)
                    logger.error(
                        f'Task failed: {task.id}, reason: {task.reason}')
                else:
                    completed.append(task.result)
                    logger.info(f'Task completed: {task.id}')

            logger.info(f'Completed sweep execution.')
            logger.info(f'Successfully completed tasks {len(completed)} tasks.')

            if failed:
                logger.warning(f'Failed {len(failed)} tasks.')

        except Exception as exception:
            logger.critical(
                f'An unexpected error occurred '
                f'in the main execution block: {exception}'
            )


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_hash',
        type=str,
        default='78ea13f7b0412909aedff5e28c0a3705c942b76086c9b7454504198c'
    )

    parser.add_argument('--budget', type=float, default=20.0)
    parser.add_argument('--network', type=str, default='polygon')

    parser.add_argument('--sweep', type=str, default='sweep.json')

    parser.add_argument('--simulation_duration', type=int, default=10000)
    parser.add_argument('--simulation_dt', type=float, default=0.1)
    parser.add_argument('--min_mem_gib', type=float, default=1.0)
    parser.add_argument('--min_cpu_threads', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=30)
    # parser.add_argument('--max_workers', type=int, default=-1)

    parser.add_argument('--logger_name', type=str, default='golem-sweep-logger')
    parser.add_argument(
        '--log_file',
        type=str,
        default=f'golem-sweep-{timestamp}.log'
    )

    config = parser.parse_args()

    enable_default_logger(log_file=config.log_file)
    logger = logging.getLogger(config.logger_name)

    try:
        asyncio.run(main(
            image_hash=config.image_hash,
            budget=config.budget,
            network=config.network,
            sweep=load_sweep_config_file(config.sweep),
            simulation_duration=config.simulation_duration,
            simulation_dt=config.simulation_dt,
            min_mem_gib=config.min_mem_gib,
            min_cpu_threads=config.min_cpu_threads,
            # timeout=config.timeout
        ))

    except KeyboardInterrupt:
        logger.info('Script interrupted by user. Shutting down gracefully...')

    except Exception as e:
        logger.critical(f'An unhandled exception occurred: {e}')
