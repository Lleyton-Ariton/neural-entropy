import datetime

import json

import itertools

import asyncio

from yapapi import Golem, Task, WorkContext

from yapapi.payload import vm
from yapapi.log import enable_default_logger

from typing import Union, Tuple, AsyncIterable


def _generate_parameterized_command(
    parameters: Tuple[Union[int, float], ...],
    duration: int,
    dt: float = 0.1,
    device: str = 'cpu',
    dtype: str = 'f32'
) -> str:
    # The base command is the interpreter and simulation script.
    command = 'python3 simulation.py'

    # Arguments for simulation script.
    args = []

    # Adding the named arguments and their values.
    args.extend(['--inputs', str(parameters[0])])
    args.extend(['--size', str(parameters[1])])
    args.extend(['--lambda', str(parameters[2])])
    args.extend(['--constant', str(parameters[3])])
    args.extend(['--excitatory_ratio', str(parameters[4])])
    args.extend(['--input_sparsity', str(parameters[5])])

    args.extend(['--input_weight_distribution_low', str(parameters[6])])
    args.extend(['--input_weight_distribution_high', str(parameters[7])])

    args.extend(
        ['--excitatory_weight_distribution_mean', str(parameters[8])]
    )
    args.extend(
        ['--excitatory_weight_distribution_deviation', str(parameters[9])]
    )

    args.extend(
        ['--inhibitory_weight_distribution_mean', str(parameters[10])]
    )
    args.extend(
        ['--inhibitory_weight_distribution_deviation', str(parameters[11])]
    )

    args.extend(['--duration', str(duration)])
    args.extend(['--dt', str(dt)])

    args.extend(['--device', device])
    args.extend(['--dtype', dtype])

    command = command + ' ' + ' '.join(args)

    return command


async def worker(context: WorkContext, tasks: AsyncIterable[Task]) -> None:
    async for task in tasks:
        script = context.new_script()

        parameters = task.data.get('parameters')
        if parameters is None:
            yield script

            task.reject_result(
                reason='No parameters were passed to the worker.'
            )

            continue

        try:
            command = _generate_parameterized_command(
                parameters=parameters,
                duration=task.data['duration'],
                dt=task.data['dt'],
                device=task.data.get('device', 'cpu'),
                dtype=task.data.get('dtype', 'f32')
            )

            future = script.run('/bin/sh', '-c', command)

            yield script

            result = await future

            if result.success:
                task.accept_result(result=result)
            else:
                task.reject_result(reason=f'Failed: {result.stderr}')

        except Exception as exception:
            task.reject_result(reason=f'Failed: {exception}')


async def main():
    package = await vm.repo(
        image_hash='709e41dddfeea628b4db3502de088d02af7e8aabc138e4250d23b2df',
        min_mem_gib=1.0
    )

    with open('sweep.json', 'r') as file:
        sweep = json.load(file)

    combinations = [*itertools.product(
        sweep['network']['inputs'], sweep['network']['size'],
        sweep['lambda'],
        sweep['constant'],
        sweep['excitatory_ratio'],
        sweep['input_sparsity'],
        sweep['inputs']['low'], sweep['inputs']['high'],
        sweep['excitatory']['means'], sweep['excitatory']['deviations'],
        sweep['inhibitory']['means'], sweep['inhibitory']['deviations']
    )]

    print('Creating tasks...')
    print(f'Created {len(combinations)} tasks.')

    # task_generator = map(
    #     lambda combination: Task(data={
    #         'parameters': combination,
    #         'duration': 100000,
    #         'dt': 0.1
    #     }), combinations)

    tasks = []
    for i, combination in enumerate(combinations):
        if i > len(combinations) - 1000:
            tasks.append(
                Task(data={
                    'parameters': combination,
                    'duration': 100000,
                    'dt': 0.1
                })
            )

    print('Starting worker pool...')

    async with Golem(
            budget=75.0, payment_network='polygon', subnet_tag='public'
    ) as golem:
        with open('results.log', 'a') as file:
            async for completed in golem.execute_tasks(
                    worker,
                    tasks,
                    payload=package,
                    # timeout=datetime.timedelta(hours=3)
            ):
                try:
                    results = json.loads(completed.result.stdout)

                    print(results)
                    file.write(f'{results}\n')
                    file.flush()

                except Exception as exception:
                    print(f'Failed: {exception}')


if __name__ == '__main__':
    # enable_default_logger(log_file='example.log')

    loop = asyncio.get_event_loop()
    task = loop.create_task(main())
    loop.run_until_complete(task)
