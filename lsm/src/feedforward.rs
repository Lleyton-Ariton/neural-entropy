use candle_core::{DType, Device, Tensor};

use crate::parameters::NeuronParameters;

pub struct Feedforward {
    pub u_t: Tensor,
    pub u_rest: Tensor,

    pub c_mem: Tensor,
    pub g_leak: Tensor,

    pub theta_t: Tensor,
    pub theta_rest: Tensor,

    pub tau_theta: Tensor,

    pub theta_reset: Tensor,

    pub g_exc: Tensor,
    pub e_exc: Tensor,

    pub tau_exc: Tensor,

    pub g_inh: Tensor,
    pub e_inh: Tensor,

    pub tau_inh: Tensor,

    pub dtype: DType
}

impl Feedforward {
    pub fn new<const N: usize>(parameters: NeuronParameters, device: &Device, dtype: DType) -> candle_core::Result<Feedforward> {
        Ok(Feedforward {
            u_t: Tensor::full(parameters.u_t, (1, N), device)?.to_dtype(dtype)?,
            u_rest: Tensor::full(parameters.u_rest, (1, N), device)?.to_dtype(dtype)?,

            c_mem: Tensor::full(parameters.c_mem, (1, N), device)?.to_dtype(dtype)?,
            g_leak: Tensor::full(parameters.g_leak, (1, N), device)?.to_dtype(dtype)?,

            theta_t: Tensor::full(parameters.theta_t, (1, N), device)?.to_dtype(dtype)?,
            theta_rest: Tensor::full(parameters.theta_rest, (1, N), device)?.to_dtype(dtype)?,

            tau_theta: Tensor::full(parameters.tau_theta, (1, N), device)?.to_dtype(dtype)?,

            theta_reset: Tensor::full(parameters.theta_reset, (1, N), device)?.to_dtype(dtype)?,

            g_exc: Tensor::full(parameters.g_exc, (1, N), device)?.to_dtype(dtype)?,
            e_exc: Tensor::full(parameters.e_exc, (1, N), device)?.to_dtype(dtype)?,

            tau_exc: Tensor::full(parameters.tau_exc, (1, N), device)?.to_dtype(dtype)?,

            g_inh: Tensor::full(parameters.g_inh, (1, N), device)?.to_dtype(dtype)?,
            e_inh: Tensor::full(parameters.e_inh, (1, N), device)?.to_dtype(dtype)?,

            tau_inh: Tensor::full(parameters.tau_inh, (1, N), device)?.to_dtype(dtype)?,

            dtype
        })
    }

    pub fn update(&mut self,
        inputs: &Tensor,
        w_inp: &Tensor,
        dt: f32
    ) -> candle_core::Result<Tensor> {
        let excitatory = inputs.matmul(&w_inp.relu()?)?;
        let inhibitory = inputs.matmul(&w_inp)?.neg()?.relu()?;

        self.g_exc = ((&self.g_exc - (&self.g_exc / &self.tau_exc)?.affine(dt as f64, 0.0)?)? + excitatory)?;
        self.g_inh = ((&self.g_inh - (&self.g_inh / &self.tau_inh)?.affine(dt as f64, 0.0)?)? + inhibitory)?;

        let i_leak = (&self.g_leak * (&self.u_rest - &self.u_t)?)?;

        let i_exc = (&self.g_exc * (&self.e_exc - &self.u_t)?)?;
        let i_inh = (&self.g_inh * (&self.e_inh - &self.u_t)?)?;

        self.u_t = (&self.u_t + ((i_leak + i_exc + i_inh)? / &self.c_mem)?.affine(dt as f64, 0.0))?;

        self.theta_t = (&self.theta_t + ((&self.theta_rest - &self.theta_t)? / &self.tau_theta)?.affine(dt as f64, 0.0))?;

        let z = self.u_t.gt(&self.theta_t)?.to_dtype(self.dtype)?;

        self.u_t = self.u_t.mul(&(1.0 - &z.detach())?)?;
        self.u_t = (&self.u_t + (&z.detach() * &self.theta_reset)?)?;

        self.theta_t = (&self.theta_t + (1.0 * &z)?)?;

        Ok(z)
    }
}