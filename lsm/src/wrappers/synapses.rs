use crate::learning::{Fixed, Synapses, STDP};

use candle_core::{DType, Device};

use pyo3::prelude::*;

#[pyclass(name = "Fixed")]
#[derive(Clone)]
pub(crate) struct FixedSynapseWrapper {
    pub inner: Fixed,
}

#[pymethods]
impl FixedSynapseWrapper {
    #[new]
    pub fn new() -> FixedSynapseWrapper {
        FixedSynapseWrapper { inner: Fixed }
    }
}

#[pyclass(name = "STDP")]
#[derive(Clone)]
pub(crate) struct STDPSynapseWrapper {
    pub inner: STDP,
}

#[pymethods]
impl STDPSynapseWrapper {
    #[new]
    #[pyo3(signature = (a_pos, a_neg, tau_pre, tau_post, shape, device = "cpu", dtype = "f32"))]
    pub fn new(
        a_pos: f32,
        a_neg: f32,
        tau_pre: f32,
        tau_post: f32,
        shape: (usize, usize),
        device: &str,
        dtype: &str,
    ) -> PyResult<STDPSynapseWrapper> {
        let device = match device {
            "cpu" => Device::Cpu,
            _ => Device::Cpu,
        };

        let dtype = match dtype {
            "f32" => DType::F32,
            "f64" => DType::F64,
            _ => DType::F32,
        };

        Ok(Self {
            inner: STDP::with(a_pos, a_neg, tau_pre, tau_post, shape, &device, dtype).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed with error: {e}"))
            })?,
        })
    }
}
