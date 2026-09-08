/*
use lsm;

use std::fs::File;

use candle_core::{DType, Device, Error, Tensor};

use itertools::iproduct;

use ndarray::{Array, ArrayBase, OwnedRepr};
use ndarray_npz::NpzWriter;

use lsm::LiquidStateMachineParameters;

use lsm::learning::{Synapses, Fixed};
use lsm::learning::{LogNormalWeightDistribution, UniformWeightDistribution};

const SECONDS: usize = 1000 * 10;

const INPUTS: usize = 1024;
const SIZE: usize = 128;

const LAMBDA: f32 = 1.0;
const CONSTANT: f32 = 0.25;

struct NetworkParameterSweep<'a> {
    inputs: &'a [usize],
    size: &'a [usize]
}

struct DistributionParameterSweep<'a> {
    means: &'a [f32],
    deviations: &'a [f32]
}

struct ParameterSweep<'a> {
    network: NetworkParameterSweep<'a>,

    lambda: &'a [f32],
    constant: &'a [f32],

    ratio: &'a [f32],

    inputs: DistributionParameterSweep<'a>,

    excitatory: DistributionParameterSweep<'a>,
    inhibitory: DistributionParameterSweep<'a>
}

const SWEEP: ParameterSweep = ParameterSweep {
    network: NetworkParameterSweep {
        inputs: &[1024],
        size: &[128]
    },

    lambda: &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    constant: &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5],

    ratio: &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

    inputs: DistributionParameterSweep {
        means: &[
            0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4
        ],
        deviations: &[1.0]
    },

    excitatory: DistributionParameterSweep {
        means: &[
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6
        ],
        deviations: &[0.2, 0.4, 0.6, 0.9, 1.2]
    },

    inhibitory: DistributionParameterSweep {
        means: &[
            0.6, 0.8, 1.2, 1.6, 1.8, 2.2, 2.4, 2.6, 2.8, 3.2
        ],
        deviations: &[0.2, 0.4, 0.6, 0.9, 1.2]
    }
};

fn main() -> Result<(), Error> {
    let parameters = iproduct!(
        iproduct!(
            SWEEP.network.inputs.iter(),
            SWEEP.network.size.iter()
        ),
        iproduct!(
            SWEEP.lambda.iter(),
            SWEEP.constant.iter(),
            SWEEP.ratio.iter()
        ),
        iproduct!(
            SWEEP.inputs.means.iter(),
            SWEEP.inputs.deviations.iter(),
            SWEEP.excitatory.means.iter(),
            SWEEP.excitatory.deviations.iter(),
            SWEEP.inhibitory.means.iter(),
            SWEEP.inhibitory.deviations.iter()
        )
    );

    let mut builder = lsm::LiquidStateMachineBuilder::new()?;

    let mut neurons = builder
        .with(LiquidStateMachineParameters::InhibitorySynapses(Synapses::Fixed(Fixed)))
        .with(LiquidStateMachineParameters::InputWeightDistribution(
            Box::new(UniformWeightDistribution::new(0.0, 1.6))
        ))
        .with(LiquidStateMachineParameters::ExcitatorySynapses(Synapses::Fixed(Fixed)))
        .with(LiquidStateMachineParameters::ExcitatoryWeightDistribution(
            Box::new(LogNormalWeightDistribution::new(0.6, 0.4))
        ))
        .with(LiquidStateMachineParameters::InhibitorySynapses(Synapses::Fixed(Fixed)))
        .with(LiquidStateMachineParameters::InhibitoryWeightDistribution(
            Box::new(LogNormalWeightDistribution::new(0.8, 0.6))
        ))
        .build::<INPUTS, SIZE>(&Device::Cpu, DType::F32)?;

    let mut recorded = Vec::new();

    let mut spikes = Tensor::zeros((1, SIZE), DType::F32, &Device::Cpu)?;
    let mut inputs = Tensor::zeros((1, INPUTS), DType::F32, &Device::Cpu)?;

    // let recordings_file = File::create_new(
    //     "../data/experiments/sweeps/sweep-1s-recordings-1.npz"
    // )?;

    // let mut recordings_writer = NpzWriter::new(recordings_file);

    let dt = 0.1;
    let rate = 40.0;

    for time in 0..(1 * SECONDS) {
        inputs = Tensor::rand(0.0, 1.0, (1, INPUTS), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .lt(&((rate / (1000.0 * 10.0)) * inputs.ones_like()?)?)?
            .to_dtype(DType::F32)?;

        spikes = neurons.step(&inputs, &spikes, dt)?;

        recorded.extend(
            spikes
                .squeeze(0)?
                .to_vec1::<f32>()?
                .iter()
                .enumerate()
                .filter_map(|(index, spike)| {
                    if *spike == 1.0 {
                        Some((time, index))
                    } else {
                        None
                    }
                })
        );
    }

    // let times = Array::from_iter(
    //     recorded.iter().map(|(time, _)| *time as f32)
    // );
    // let indices = Array::from_iter(
    //     recorded.iter().map(|(_, index)| *index as f32)
    // );

    // recordings_writer.add_array("times", &times).unwrap();
    // recordings_writer.add_array("spikes", &indices).unwrap();

    // recordings_writer.finish().unwrap();

    Ok(())
}
*/

fn main() {

}