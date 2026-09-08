import sys
import uuid

import json

import argparse

import lsm


def run(config: argparse.Namespace, trials: int = 1) -> None:
    try:
        for i in range(trials):
            # Network setup and configuration.
            network = lsm.LiquidStateMachine(
                **{
                    'inputs': config.inputs,
                    'size': config.size,
                    'lambda': config.lambda_,
                    'constant': config.constant,
                    'excitatory_ratio': config.excitatory_ratio,
                    'input_sparsity': config.input_sparsity,
                    'input_synapses': lsm.Fixed(),
                    'input_weight_distribution':
                        lsm.UniformWeightDistribution(
                            config.input_weight_distribution_low,
                            config.input_weight_distribution_high
                        ),
                    'excitatory_synapses': lsm.Fixed(),
                    'excitatory_weight_distribution':
                        lsm.LogNormalWeightDistribution(
                            config.excitatory_weight_distribution_mean,
                            config.excitatory_weight_distribution_deviation
                        ),
                    'inhibitory_synapses': lsm.Fixed(),
                    'inhibitory_weight_distribution':
                        lsm.LogNormalWeightDistribution(
                            config.inhibitory_weight_distribution_mean,
                            config.inhibitory_weight_distribution_deviation
                        ),
                },
                device=config.device,
                dtype=config.dtype
            )

            # The network will return a list of tuples.
            results = network.run(config.duration, dt=config.dt)

            # Write the results to stdout so as to be captured by the requester.
            sys.stdout.write(
                json.dumps({
                    'id': str(uuid.uuid4()),
                    'parameters': {
                        'inputs': config.inputs,
                        'size': config.size,
                        'lambda': config.lambda_,
                        'constant': config.constant,
                        'excitatory_ratio': config.excitatory_ratio,
                        'input_sparsity': config.input_sparsity,
                        'input_weight_distribution': {
                            'low': config.input_weight_distribution_low,
                            'high': config.input_weight_distribution_high
                        },
                        'excitatory_weight_distribution': {
                            'mean': config.excitatory_weight_distribution_mean,
                            'deviation':
                                config.excitatory_weight_distribution_deviation
                        },
                        'inhibitory_weight_distribution': {
                            'mean': config.inhibitory_weight_distribution_mean,
                            'deviation':
                                config.inhibitory_weight_distribution_deviation
                        }
                    },
                    'recordings': {
                        'trial': i,
                        'results': len(results)
                    }
                })
            )

    except Exception as exception:
        # Print a meaningful error message to stderr for debugging.
        print(f'Error: {exception}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputs', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--lambda', dest='lambda_', type=float, required=True)
    parser.add_argument('--constant', type=float, required=True)
    parser.add_argument('--excitatory_ratio', type=float, required=True)
    parser.add_argument('--input_sparsity', type=float, required=True)
    parser.add_argument(
        '--input_weight_distribution_low', type=float, required=True
    )
    parser.add_argument(
        '--input_weight_distribution_high', type=float, required=True
    )
    parser.add_argument(
        '--excitatory_weight_distribution_mean', type=float, required=True
    )
    parser.add_argument(
        '--excitatory_weight_distribution_deviation', type=float, required=True
    )
    parser.add_argument(
        '--inhibitory_weight_distribution_mean', type=float, required=True
    )
    parser.add_argument(
        '--inhibitory_weight_distribution_deviation', type=float, required=True
    )

    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--dt', type=float, required=True)

    parser.add_argument('--device', type=str, required=False, default='cpu')
    parser.add_argument('--dtype', type=str, required=False, default='f32')

    parser.add_argument('--trials', type=int, required=False, default=1)

    run(config=parser.parse_args())
