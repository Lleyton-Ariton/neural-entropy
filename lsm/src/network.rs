use candle_core::{DType, Device, Tensor};

use crate::recurrent::Recurrent;

use crate::learning::{Synapses, STDP, Fixed};
use crate::learning::{WeightDistribution, UniformWeightDistribution};

pub enum LiquidStateMachineParameters {
    DistanceLambda(f32),
    DistanceConstant(f32),
    ExcitatoryRatio(f32),
    InputSparsity(f32),
    InputSynapses(Synapses),
    InputWeightDistribution(Box<dyn WeightDistribution<(usize, usize)>>),
    ExcitatorySynapses(Synapses),
    ExcitatoryWeightDistribution(Box<dyn WeightDistribution<(usize, usize)>>),
    InhibitorySynapses(Synapses),
    InhibitoryWeightDistribution(Box<dyn WeightDistribution<(usize, usize)>>)
}

pub struct LiquidStateMachineBuilder {
    lambda: f32,
    constant: f32,

    excitatory_ratio: f32,

    input_sparsity: f32,

    w_inp_synapses: Synapses,
    w_inp_distribution: Box<dyn WeightDistribution<(usize, usize)>>,

    w_rec_exc_synapses: Synapses,
    w_rec_exc_distribution: Box<dyn WeightDistribution<(usize, usize)>>,

    w_rec_inh_synapses: Synapses,
    w_rec_inh_distribution: Box<dyn WeightDistribution<(usize, usize)>>
}

impl LiquidStateMachineBuilder {
    pub fn new() -> candle_core::Result<LiquidStateMachineBuilder> {
        Ok(LiquidStateMachineBuilder {
            lambda: 1.0,
            constant: 0.25,

            excitatory_ratio: 0.8,

            input_sparsity: 0.1,

            w_inp_synapses: Synapses::Fixed(Fixed),
            w_inp_distribution: Box::new(UniformWeightDistribution::new(0.0, 7.0)),

            w_rec_exc_synapses: Synapses::Fixed(Fixed),
            w_rec_exc_distribution: Box::new(UniformWeightDistribution::new(0.0, 3.5)),

            w_rec_inh_synapses: Synapses::Fixed(Fixed),
            w_rec_inh_distribution: Box::new(UniformWeightDistribution::new(-4.6, 0.0))
        })
    }

    pub fn with(self, parameter: LiquidStateMachineParameters) -> LiquidStateMachineBuilder {
        match parameter {
            LiquidStateMachineParameters::DistanceLambda(lambda) => {
                LiquidStateMachineBuilder {
                    lambda,

                    ..self
                }
            }
            LiquidStateMachineParameters::DistanceConstant(constant) => {
                LiquidStateMachineBuilder {
                    constant,

                    ..self
                }
            }
            LiquidStateMachineParameters::ExcitatoryRatio(excitatory_ratio) => {
                LiquidStateMachineBuilder {
                    excitatory_ratio,

                    ..self
                }
            }
            LiquidStateMachineParameters::InputSparsity(input_sparsity) => {
                LiquidStateMachineBuilder {
                    input_sparsity,

                    ..self
                }
            }
            LiquidStateMachineParameters::InputSynapses(w_inp_synapses) => {
                LiquidStateMachineBuilder {
                    w_inp_synapses,

                    ..self
                }
            }
            LiquidStateMachineParameters::InputWeightDistribution(w_inp_distribution) => {
                LiquidStateMachineBuilder {
                    w_inp_distribution,

                    ..self
                }
            }
            LiquidStateMachineParameters::ExcitatorySynapses(w_rec_exc_synapses) => {
                LiquidStateMachineBuilder {
                    w_rec_exc_synapses,

                    ..self
                }
            }
            LiquidStateMachineParameters::ExcitatoryWeightDistribution(w_rec_exc_distribution) => {
                LiquidStateMachineBuilder {
                    w_rec_exc_distribution,

                    ..self
                }
            }
            LiquidStateMachineParameters::InhibitorySynapses(w_rec_inh_synapses) => {
                LiquidStateMachineBuilder {
                    w_rec_inh_synapses,

                    ..self
                }
            }
            LiquidStateMachineParameters::InhibitoryWeightDistribution(w_rec_inh_distribution) => {
                LiquidStateMachineBuilder {
                    w_rec_inh_distribution,

                    ..self
                }
            }
        }
    }

    pub fn build<const N: usize, const M: usize>(mut self, device: &Device, dtype: DType) -> candle_core::Result<LiquidStateMachine> {
        LiquidStateMachineBuilder::build_with_dimensions(self, N, M, device, dtype)
    }

    pub fn build_with_dimensions(mut self, n: usize, m: usize, device: &Device, dtype: DType) -> candle_core::Result<LiquidStateMachine> {
        let lambda = Tensor::full(self.lambda, (m,), device)?.to_dtype(dtype)?;
        let constant = Tensor::full(self.constant, (m,), device)?.to_dtype(dtype)?;

        let coordinates = Tensor::rand(0.0, 1.0, (m, 3), device)?.to_dtype(dtype)?;

        let mut distances = Vec::new();

        for i in 0..m {
            let euclidean = (&coordinates - coordinates.narrow(0, i, 1)?.repeat((m,))?)?
                .powf(2.0)?
                .sum(1)?;

            distances.push((&constant * (euclidean / &lambda)?.powf(2.0)?.neg()?.exp()?)?);
        }

        let inp_connections = Tensor::rand(0.0, 1.0, (n, m), device)?.to_dtype(dtype)?;
        let inp_connections = inp_connections
            .lt(&Tensor::full(0.10, (n, m), device)?.to_dtype(dtype)?)?
            .to_dtype(dtype)?;

        let rec_connections = Tensor::rand(0.0, 1.0, (m, m), device)?.to_dtype(dtype)?;
        let rec_connections = rec_connections
            .lt(&Tensor::stack(&distances, 0)?)?
            .to_dtype(dtype)?;

        let rec_connections = (rec_connections * (1.0 - Tensor::eye(m, dtype, device)?)?)?;

        let scale = 1.0 / (m as f64).sqrt();

        let w_exc = self.w_rec_exc_distribution.sample((m, m), device)?.affine(scale, 0.0)?.to_dtype(dtype)?;
        let w_inh = self.w_rec_inh_distribution.sample((m, m), device)?.affine(scale, 0.0)?.to_dtype(dtype)?;

        let excitatory = Tensor::rand(0.0, 1.0, (m, 1), device)?
            .to_dtype(dtype)?
            .lt(&Tensor::full(self.excitatory_ratio, (m, 1), device)?)?
            .to_dtype(dtype)?;

        let inhibitory = (Tensor::ones((m, 1), dtype, device)? - &excitatory)?;

        let w_exc = (w_exc * excitatory.repeat((1, m))?)?;
        let w_inh = (w_inh * inhibitory.repeat((1, m))?)?;

        let w_inp = self.w_inp_distribution.sample((n, m), device)?.affine(scale, 0.0)?.to_dtype(dtype)?;
        let w_inp = (w_inp * &inp_connections)?;

        let w_rec = ((w_exc + w_inh)? * &rec_connections)?;

        Ok(LiquidStateMachine {
            liquid: Recurrent::with(m, device, dtype)?,

            inp_connections,
            rec_connections,

            exc_connections: excitatory.repeat((1, m))?,
            inh_connections: inhibitory.repeat((1, m))?,

            w_inp,
            w_rec,

            w_inp_learner: self.w_inp_synapses,

            w_rec_exc_learner: self.w_rec_exc_synapses,
            w_rec_inh_learner: self.w_rec_inh_synapses
        })
    }
}

pub struct LiquidStateMachine {
    liquid: Recurrent,

    inp_connections: Tensor,
    rec_connections: Tensor,

    exc_connections: Tensor,
    inh_connections: Tensor,

    w_inp: Tensor,
    w_rec: Tensor,

    w_inp_learner: Synapses,

    w_rec_exc_learner: Synapses,
    w_rec_inh_learner: Synapses
}

impl LiquidStateMachine {
    pub fn new<const N: usize, const M: usize>(
        device: &Device,
        dtype: DType
    ) -> candle_core::Result<LiquidStateMachine> {
        LiquidStateMachineBuilder::new()?.build::<N, M>(device, dtype)
    }

    pub fn step(&mut self, inputs: &Tensor, spikes: &Tensor, dt: f32) -> candle_core::Result<Tensor> {
        let z = self.liquid.update(inputs, spikes, &self.w_inp, &self.w_rec, dt)?;

        let updated_w_inp = self.w_inp_learner.update(&self.w_inp, inputs, &z, dt)?.clamp(0.0, 10.0)?;

        self.w_inp = (updated_w_inp * &self.inp_connections)?;

        let updated_w_rec_exc = self.w_rec_exc_learner.update(&self.w_rec, spikes, &z, dt)?.clamp(0.0, 10.0)?;
        let updated_w_rec_inh = self.w_rec_inh_learner.update(&self.w_rec, spikes, &z, dt)?.clamp(-10.0, 0.0)?;

        let w_rec_exc = (updated_w_rec_exc * &self.exc_connections)?;
        let w_rec_inh = (updated_w_rec_inh * &self.inh_connections)?;

        self.w_rec = ((&w_rec_exc + &w_rec_inh)? * &self.rec_connections)?;

        Ok(z)
    }

    pub fn advance(&mut self, inputs: &Tensor, spikes: &Tensor, dt: f32) -> candle_core::Result<Tensor> {
        let z = self.liquid.update(inputs, spikes, &self.w_inp, &self.w_rec, dt)?;

        Ok(z)
    }

    pub fn weights(&self) -> (&Tensor, &Tensor) {
        (&self.w_inp, &self.w_rec)
    }
}
