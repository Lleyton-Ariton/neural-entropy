import csv
import json

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import lsm

from typing import Union, Dict, List


PARAMETERS_FILE = './data/recordings-1s-shannon-entropy-parameters.csv'
ENTROPY_FILE = './data/recordings-1s-shannon-entropy-values.csv'


def load_recordings_parameter_data(
        file_name: str
) -> Dict[str, Dict[str, Union[int, float]]]:
    parameters = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file.readlines(), delimiter=',')

        for row in reader:
            id_, parameter_name, parameter_choice = row[1], row[2], int(row[3])

            parameter_values = parameters.get(id_, {})
            parameter_values.update({
                parameter_name: json.loads(
                    row[-1]
                )['attributes']['choices'][parameter_choice]
            })

            parameters[id_] = parameter_values

    return parameters


def load_recordings_entropy_data(file_name: str) -> Dict[str, List[float]]:
    data = {}

    with open(file_name, 'r') as file:
        for line in file.readlines():
            _, id_, trial, entropy, status = line.strip('\n').split(',')

            if status == 'NAN':
                continue

            values = data.get(id_, [])
            values.append(float(entropy))

            data[id_] = values

    return data


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42

    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6

    sweeps = load_recordings_parameter_data(PARAMETERS_FILE)
    entropies = load_recordings_entropy_data(ENTROPY_FILE)

    w, x, y, z = [], [], [], []

    for key, items in sweeps.items():
        topological_entropy = entropies.get(key, None)

        if topological_entropy is not None:
            w.append(
                items['inhibitory_weight_distribution_mean'] /
                items['excitatory_weight_distribution_mean']
            )
            x.append(
                lsm.LiquidStateMachine(
                    **{
                        'lambda': items['lambda'],
                        'constant': items['constant']
                    }
                ).sparsity()
            )
            y.append(items['excitatory_ratio'])
            z.append(
                sum(topological_entropy) / len(topological_entropy)
            )

    w = np.array(w)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fig = plt.figure(figsize=(7.08, 6.7))
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(True, linestyle='--', alpha=0.25)

    scatter = ax.scatter(
        x, y, z, c=z, cmap='magma', marker='o', s=50
    )

    maximum = np.argmax(z)

    zs = np.linspace(0.0, np.max(z), 100)
    xs, zsg1 = np.meshgrid(np.linspace(0.0, 1.0, 100), zs)
    ys, zsg2 = np.meshgrid(np.linspace(0.0, 1.0, 100), zs)

    ax.plot_surface(
        xs, y[maximum] * np.ones_like(xs), zsg1, color='grey', alpha=0.15
    )
    ax.plot_surface(
        x[maximum] * np.ones_like(ys), ys, zsg2, color='grey', alpha=0.15
    )

    ax.plot(
        x[maximum], y[maximum], z[maximum], marker='+', color='black', alpha=1.0
    )

    x_label = ax.set_xlabel('Sparsity')
    y_label = ax.set_ylabel('Excitatory Ratio')
    z_label = ax.set_zlabel('Entropy')
    
    ax.view_init(elev=25, azim=-45)

    plt.savefig(
        'stationary-distribution-shannon-entropy.pdf',
        dpi=600,
        format='pdf',
        bbox_inches='tight',
        bbox_extra_artists=[x_label, y_label, z_label]
    )
