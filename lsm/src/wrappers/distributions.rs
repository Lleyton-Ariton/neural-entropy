use candle_core::{Device, Tensor};

use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;

use crate::learning::{
    WeightDistribution,
    UniformWeightDistribution,
    NormalWeightDistribution,
    LogNormalWeightDistribution,
};

#[pyclass(name = "UniformWeightDistribution")]
#[derive(Clone)]
pub struct UniformWeightDistributionWrapper {
    pub distribution: UniformWeightDistribution,
}

#[pymethods]
impl UniformWeightDistributionWrapper {
    #[new]
    pub fn new(low: f32, high: f32) -> PyResult<UniformWeightDistributionWrapper> {
        Ok(Self {
            distribution: UniformWeightDistribution::new(low, high),
        })
    }

    pub fn low(&self) -> PyResult<f32> {
        Ok(self.distribution.low)
    }

    pub fn high(&self) -> PyResult<f32> {
        Ok(self.distribution.high)
    }
}

impl WeightDistribution<(usize, usize)> for UniformWeightDistributionWrapper {
    fn sample(&mut self, shape: (usize, usize), device: &Device) -> candle_core::Result<Tensor> {
        self.distribution.sample(shape, device)
    }
}

#[pyclass(name = "NormalWeightDistribution")]
#[derive(Clone)]
pub struct NormalWeightDistributionWrapper {
    pub distribution: NormalWeightDistribution,
}

#[pymethods]
impl NormalWeightDistributionWrapper {
    #[new]
    pub fn new(loc: f32, scale: f32) -> PyResult<NormalWeightDistributionWrapper> {
        Ok(Self {
            distribution: NormalWeightDistribution::new(loc, scale),
        })
    }

    pub fn loc(&self) -> PyResult<f32> {
        Ok(self.distribution.loc)
    }

    pub fn scale(&self) -> PyResult<f32> {
        Ok(self.distribution.scale)
    }
}

impl WeightDistribution<(usize, usize)> for NormalWeightDistributionWrapper {
    fn sample(&mut self, shape: (usize, usize), device: &Device) -> candle_core::Result<Tensor> {
        self.distribution.sample(shape, device)
    }
}

#[pyclass(name = "LogNormalWeightDistribution")]
#[derive(Clone)]
pub struct LogNormalWeightDistributionWrapper {
    pub distribution: LogNormalWeightDistribution,
}

#[pymethods]
impl LogNormalWeightDistributionWrapper {
    #[new]
    pub fn new(loc: f32, scale: f32) -> PyResult<LogNormalWeightDistributionWrapper> {
        Ok(Self {
            distribution: LogNormalWeightDistribution::new(loc, scale),
        })
    }

    pub fn loc(&self) -> PyResult<f32> {
        Ok(self.distribution.loc)
    }

    pub fn scale(&self) -> PyResult<f32> {
        Ok(self.distribution.scale)
    }
}

impl WeightDistribution<(usize, usize)> for LogNormalWeightDistributionWrapper {
    fn sample(&mut self, shape: (usize, usize), device: &Device) -> candle_core::Result<Tensor> {
        self.distribution.sample(shape, device)
    }
}
