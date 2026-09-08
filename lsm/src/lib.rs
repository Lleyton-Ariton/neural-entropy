pub mod parameters;

pub mod feedforward;
pub mod recurrent;

pub mod learning;

pub mod network;

mod wrappers;

pub use {
    feedforward::Feedforward,
    network::{LiquidStateMachine, LiquidStateMachineBuilder, LiquidStateMachineParameters},
    parameters::NeuronParameters,
    recurrent::Recurrent,
};

use learning::{Fixed, Synapses, UniformWeightDistribution, WeightDistribution};

use wrappers::synapses::{FixedSynapseWrapper, STDPSynapseWrapper};
use wrappers::distributions::{
    LogNormalWeightDistributionWrapper,
    NormalWeightDistributionWrapper,
    UniformWeightDistributionWrapper
};

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::types::PyList;

use candle_core::{DType, Device, Tensor};

#[pyclass(name = "LiquidStateMachine")]
struct LiquidStateMachineWrapper {
    inputs: usize,
    size: usize,

    lsm: LiquidStateMachine,

    device: Device,
    dtype: DType
}

#[pymethods]
impl LiquidStateMachineWrapper {
    #[new]
    #[pyo3(signature = (
    inputs = 256,
    size = 128,
    lambda = 1.0,
    constant = 0.25,
    excitatory_ratio = 0.8,
    input_sparsity = 0.1,
    input_synapses = None,
    input_weight_distribution = None,
    excitatory_synapses = None,
    excitatory_weight_distribution = None,
    inhibitory_synapses = None,
    inhibitory_weight_distribution = None,
    device = "cpu",
    dtype = "f32"
    ))]
    pub fn new(
        inputs: usize,
        size: usize,
        lambda: f32,
        constant: f32,
        excitatory_ratio: f32,
        input_sparsity: f32,
        input_synapses: Option<Py<PyAny>>,
        input_weight_distribution: Option<Py<PyAny>>,
        excitatory_synapses: Option<Py<PyAny>>,
        excitatory_weight_distribution: Option<Py<PyAny>>,
        inhibitory_synapses: Option<Py<PyAny>>,
        inhibitory_weight_distribution: Option<Py<PyAny>>,
        device: &str,
        dtype: &str,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let input_synapses = match input_synapses {
            Some(obj) => extract_synapse_type(obj.bind(py))?,
            None => Synapses::Fixed(Fixed)
        };
        let excitatory_synapses = match excitatory_synapses {
            Some(obj) => extract_synapse_type(obj.bind(py))?,
            None => Synapses::Fixed(Fixed)
        };
        let inhibitory_synapses = match inhibitory_synapses {
            Some(obj) => extract_synapse_type(obj.bind(py))?,
            None => Synapses::Fixed(Fixed)
        };

        let input_weight_distribution = match input_weight_distribution {
            Some(obj) => extract_weight_distribution_type(obj.bind(py))?,
            None => Box::new(UniformWeightDistributionWrapper::new(0.0, 7.0)?),
        };
        let excitatory_weight_distribution = match excitatory_weight_distribution {
            Some(obj) => extract_weight_distribution_type(obj.bind(py))?,
            None => Box::new(UniformWeightDistributionWrapper::new(0.0, 3.5)?)
        };
        let inhibitory_weight_distribution = match inhibitory_weight_distribution {
            Some(obj) => extract_weight_distribution_type(obj.bind(py))?,
            None => Box::new(UniformWeightDistributionWrapper::new(-4.6, 0.0)?)
        };

        let device = match device {
            "cpu" => Device::Cpu,
            other if other.starts_with("cuda") => Device::new_cuda(0)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed with error: {e}")))?,
            _ => Device::Cpu,
        };
        let dtype = match dtype {
            "f32" => DType::F32,
            "f64" => DType::F64,
            _ => DType::F32,
        };

        let mut builder = LiquidStateMachineBuilder::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed with error: {e}")))?;

        builder = builder
            .with(LiquidStateMachineParameters::DistanceLambda(lambda))
            .with(LiquidStateMachineParameters::DistanceConstant(constant))
            .with(LiquidStateMachineParameters::ExcitatoryRatio(
                excitatory_ratio,
            ))
            .with(LiquidStateMachineParameters::InputSparsity(input_sparsity))
            .with(LiquidStateMachineParameters::InputSynapses(input_synapses))
            .with(LiquidStateMachineParameters::InputWeightDistribution(
                input_weight_distribution,
            ))
            .with(LiquidStateMachineParameters::ExcitatorySynapses(
                excitatory_synapses,
            ))
            .with(LiquidStateMachineParameters::ExcitatoryWeightDistribution(
                excitatory_weight_distribution,
            ))
            .with(LiquidStateMachineParameters::InhibitorySynapses(
                inhibitory_synapses,
            ))
            .with(LiquidStateMachineParameters::InhibitoryWeightDistribution(
                inhibitory_weight_distribution,
            ));

        let lsm = builder
            .build_with_dimensions(inputs, size, &device, dtype)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed with error: {e}")))?;

        Ok(LiquidStateMachineWrapper {
            inputs,
            size,

            lsm,

            device,
            dtype
        })
    }

    pub fn weights(&self) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let weights = self.lsm.weights();

        let weights = (|| {
            Ok((
                weights.0.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                weights.1.to_dtype(DType::F32)?.to_vec2::<f32>()?
            ))
        } )();

        Ok(weights.map_err(|e: candle_core::Error| PyRuntimeError::new_err(format!("Failed with error: {e}")))?)
    }

    pub fn sparsity(&self) -> PyResult<f32> {
        let (_, weights) = self.lsm.weights();

        let weight_dims = weights.dims();
        let (n, m) = (weight_dims[0], weight_dims[1]);

        let sparsity: Result<f32, candle_core::Error> = (|| {
            Ok(weights
                .gt(&weights.zeros_like()?)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_vec0::<f32>()? / ((n * m) as f32))
        } )();

        Ok(sparsity.map_err(|e| PyRuntimeError::new_err(format!("Failed with error: {e}")))?)
    }

    #[pyo3(signature = (duration, rate = 40.0, dt = 0.1))]
    pub fn run(
        &mut self,
        duration: usize,
        rate: f32,
        dt: f32
    ) -> PyResult<Vec<(usize, usize)>> {
        let mut recorded = Vec::new();

        let result: candle_core::Result<Vec<(usize, usize)>> = (|| {
            let mut inputs = Tensor::zeros((1, self.inputs), self.dtype, &self.device)?;
            let mut spikes = Tensor::zeros((1, self.size), self.dtype, &self.device)?;

            for time in 0..duration {
                inputs = Tensor::rand(0.0, 1.0, (1, self.inputs), &self.device)?
                    .to_dtype(DType::F32)?
                    .lt(&inputs.ones_like()?.affine((rate / (1000.0 * 10.0)) as f64, 0.0)?)?
                    .to_dtype(DType::F32)?;

                spikes = self.lsm.step(&inputs, &spikes, dt)?;

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

            Ok(recorded)
        })();

        Ok(result.map_err(|e| PyRuntimeError::new_err(format!("Failed with error: {e}")))?)
    }
}

fn extract_synapse_type(
    ob: &Bound<'_, PyAny>,
) -> PyResult<Synapses> {
    if let Ok(fixed_wrapper) = ob.extract::<FixedSynapseWrapper>() {
        return Ok(Synapses::Fixed(fixed_wrapper.inner.clone()));
    }

    if let Ok(stdp_wrapper) = ob.extract::<STDPSynapseWrapper>() {
        return Ok(Synapses::STDP(stdp_wrapper.inner.clone()));
    }

    Err(PyTypeError::new_err(
        "Received an unknown synaptic type, currently only either Fixed' or 'STDP' are supported."
    ))
}

fn extract_weight_distribution_type(
    ob: &Bound<'_, PyAny>,
) -> PyResult<Box<dyn WeightDistribution<(usize, usize)>>> {
    if let Ok(uniform_wrapper) = ob.extract::<PyRef<UniformWeightDistributionWrapper>>() {
        return Ok(Box::new(uniform_wrapper.distribution.clone()));
    }

    if let Ok(normal_wrapper) = ob.extract::<PyRef<NormalWeightDistributionWrapper>>() {
        return Ok(Box::new(normal_wrapper.distribution.clone()));
    }

    if let Ok(log_normal_wrapper) = ob.extract::<PyRef<LogNormalWeightDistributionWrapper>>() {
        return Ok(Box::new(log_normal_wrapper.distribution.clone()));
    }

    Err(PyTypeError::new_err(
        "Received an unknown distribution, currently only either \
        'UniformWeightDistribution', 'NormalWeightDistribution' or 'LogNormalWeightDistribution' are supported."
    ))
}

#[pymodule]
fn lsm(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FixedSynapseWrapper>()?;
    module.add_class::<STDPSynapseWrapper>()?;

    module.add_class::<UniformWeightDistributionWrapper>()?;
    module.add_class::<NormalWeightDistributionWrapper>()?;
    module.add_class::<LogNormalWeightDistributionWrapper>()?;

    module.add_class::<LiquidStateMachineWrapper>()?;

    Ok(())
}
