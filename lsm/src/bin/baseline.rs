use lsm;

use std::fs::File;

use candle_core::{DType, Device, Error, Tensor};

use ndarray::{Array, ArrayBase, OwnedRepr};
use ndarray_npz::NpzWriter;

use plotly::common::Mode::Markers;
use plotly::{Plot, Scatter};

const SECONDS: usize = 1000 * 10;

const INPUTS: usize = 256;
const SIZE: usize = 128;

const LAMBDA: f32 = 1.0;
const CONSTANT: f32 = 0.25;

fn main() -> Result<(), Error> {
    let mut neurons = lsm::LiquidStateMachine::new::<INPUTS, SIZE>(&Device::Cpu, DType::F32)?;

    let mut recorded = Vec::new();

    let mut spikes = Tensor::zeros((1, SIZE), DType::F32, &Device::Cpu)?;
    let mut inputs = Tensor::zeros((1, INPUTS), DType::F32, &Device::Cpu)?;

    let recordings_file = File::create_new(
        "../data/experiments/baseline/baseline-spikes-1s-recording.npz"
    )?;

    let mut recordings_writer = NpzWriter::new(recordings_file);

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

    let times = Array::from_iter(
        recorded.iter().map(|(time, _)| *time as f32)
    );
    let indices = Array::from_iter(
        recorded.iter().map(|(_, index)| *index as f32)
    );

    recordings_writer.add_array("times", &times).unwrap();
    recordings_writer.add_array("spikes", &indices).unwrap();

    recordings_writer.finish().unwrap();

    Ok(())
}
