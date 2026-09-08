import collections

from yapapi import Task, WorkContext

from typing import (
    Optional, Union, List, Tuple
)

import lsm


# Simulation execution function.
def run_simulation(
        parameters: Tuple[Union[int, float], ...],
        duration: int,
        dt: Optional[float] = 0.1,
        device: Optional[str] = 'cpu',
        dtype: Optional[str] = 'f32'
) -> List[Tuple[int, int]]:
    try:
        network = lsm.LiquidStateMachine(
            **{
                'inputs': parameters[0],
                'size': parameters[1],
                'lambda': parameters[2],
                'constant': parameters[3],
                'excitatory_ratio': parameters[4],
                'input_sparsity': parameters[5],
                'input_synapses': lsm.Fixed(),
                'input_weight_distribution':
                    lsm.UniformWeightDistribution(
                        parameters[6], parameters[7]
                    ),
                'excitatory_synapses': lsm.Fixed(),
                'excitatory_weight_distribution':
                    lsm.LogNormalWeightDistribution(
                        parameters[8], parameters[9]
                    ),
                'inhibitory_synapses': lsm.Fixed(),
                'inhibitory_weight_distribution':
                    lsm.LogNormalWeightDistribution(
                        parameters[10], parameters[11]
                    ),
            },
            device=device,
            dtype=dtype
        )

        # The network will return a list of tuples.
        return network.run(duration, dt=dt)

    # If any exception occurs, raise and exit the function.
    # All exceptions should be caught and handled by the worker.
    except Exception as _:
        raise


# Worker function which executes a task on the provider side.
async def worker(
        context: WorkContext,
        tasks: collections.AsyncIterable[Task]
) -> None:
    async for task in tasks:
        script = context.new_script()

        parameters = task.data.get('parameters')
        if parameters is None:
            task.reject_result(
                reason='No parameters were passed to the worker.'
            )
            yield script
            continue

        try:
            simulation_result = run_simulation(
                parameters,
                duration=task.data['duration'],
                dt=task.data['dt']
            )

            task.accept_result(result=simulation_result)
            print(f'Task completed successfully for parameters: {parameters}')

        except Exception as exception:
            task.reject_result(reason=f'Worker failed: {exception}')
            print(
                f'Task failed on provider '
                f'for parameters {parameters}: {exception}'
            )

        yield script
