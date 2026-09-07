use candle_core::{DType, Device, Tensor, Shape};

#[derive(Clone)]
pub struct Fixed;

impl Fixed {
    pub fn update<T1, T2, T3>(&mut self, w: &Tensor, _: T1, _: T2, _: T3) -> candle_core::Result<Tensor> {
        let w = w.affine(1.0, 0.0)?;

        Ok(w)
    }
}

#[derive(Clone)]
pub struct STDP {
    pub a_pos: Tensor,
    pub a_neg: Tensor,

    pub trace_pre: Tensor,
    pub trace_post: Tensor,

    pub tau_pre: Tensor,
    pub tau_post: Tensor,
}

impl STDP {
    pub fn new<const N: usize, const M: usize>(
        a_pos: f32,
        a_neg: f32,
        tau_pre: f32,
        tau_post: f32,
        device: &Device,
        dtype: DType
    ) -> candle_core::Result<STDP> {
        Ok(STDP {
            a_pos: Tensor::full(a_pos, (M, N), device)?.to_dtype(dtype)?,
            a_neg: Tensor::full(a_neg, (M, N), device)?.to_dtype(dtype)?,

            trace_pre: Tensor::zeros((1, N), dtype, device)?.to_dtype(dtype)?,
            trace_post: Tensor::zeros((1, M), dtype, device)?.to_dtype(dtype)?,

            tau_pre: Tensor::full(tau_pre, (1, N), device)?.to_dtype(dtype)?,
            tau_post: Tensor::full(tau_post, (1, M), device)?.to_dtype(dtype)?,
        })
    }

    pub fn with(
        a_pos: f32,
        a_neg: f32,
        tau_pre: f32,
        tau_post: f32,
        shape: (usize, usize),
        device: &Device,
        dtype: DType
    ) -> candle_core::Result<STDP> {
        let (n, m) = (shape.0, shape.1);

        Ok(STDP {
            a_pos: Tensor::full(a_pos, (m, n), device)?.to_dtype(dtype)?,
            a_neg: Tensor::full(a_neg, (m, n), device)?.to_dtype(dtype)?,

            trace_pre: Tensor::zeros((1, n), dtype, device)?.to_dtype(dtype)?,
            trace_post: Tensor::zeros((1, m), dtype, device)?.to_dtype(dtype)?,

            tau_pre: Tensor::full(tau_pre, (1, n), device)?.to_dtype(dtype)?,
            tau_post: Tensor::full(tau_post, (1, m), device)?.to_dtype(dtype)?,
        })
    }

    pub fn update(
        &mut self,
        w: &Tensor,
        pre: &Tensor,
        post: &Tensor,
        dt: f32
    ) -> candle_core::Result<Tensor> {
        self.trace_pre =
            (&self.trace_pre -
                (&self.trace_pre / &self.tau_pre)?.affine(dt as f64, 0.0)? +
                (pre / &self.tau_pre)?.affine(dt as f64, 0.0)?
            )?;
        self.trace_post =
            (&self.trace_post -
                (&self.trace_post / &self.tau_post)?.affine(dt as f64, 0.0)? -
                (post / &self.tau_post)?.affine(dt as f64, 0.0)?
            )?;

        let delta_w_pre = (&self.a_pos
            * &self
                .trace_post
                .unsqueeze(2)?
                .broadcast_mul(&pre.unsqueeze(1)?)?
                .sum(0)?)?;
        let delta_w_post = (&self.a_neg
            * &self
                .trace_pre
                .unsqueeze(1)?
                .broadcast_mul(&post.unsqueeze(2)?)?
                .sum(0)?)?;

        let delta_w = (delta_w_pre + delta_w_post)?.t()?;

        let w = (w + delta_w)?;

        Ok(w)
    }
}

#[derive(Clone)]
pub enum Synapses {
    Fixed(Fixed),
    STDP(STDP),
}

impl Synapses {
    pub fn update(&mut self, w: &Tensor, pre: &Tensor, post: &Tensor, dt: f32) -> candle_core::Result<Tensor> {
        match self {
            Synapses::STDP(learner) => learner.update(w, pre, post, dt),
            Synapses::Fixed(learner) => learner.update(w, pre, post, dt)
        }
    }
}

pub trait WeightDistribution<S: Into<Shape>> {
    fn sample(&mut self, shape: S, device: &Device) -> candle_core::Result<Tensor>;
}


#[derive(Copy, Clone)]
pub struct UniformWeightDistribution {
    pub low: f32,
    pub high: f32
}

impl UniformWeightDistribution {
    pub fn new(low: f32, high: f32) -> UniformWeightDistribution {
        UniformWeightDistribution { 
            low, 
            high 
        }
    }
}

#[derive(Copy, Clone)]
pub struct NormalWeightDistribution {
    pub loc: f32,
    pub scale: f32
}

impl NormalWeightDistribution {
    pub fn new(loc: f32, scale: f32) -> NormalWeightDistribution {
        NormalWeightDistribution {
            loc,
            scale
        }
    }
}

impl WeightDistribution<(usize, usize)> for NormalWeightDistribution {
    fn sample(&mut self, shape: (usize, usize), device: &Device) -> candle_core::Result<Tensor> {
        Tensor::randn(self.loc, self.scale, shape, device)
    }
}

impl WeightDistribution<(usize, usize)> for UniformWeightDistribution {
    fn sample(&mut self, shape: (usize, usize), device: &Device) -> candle_core::Result<Tensor> {
        Tensor::rand(self.low, self.high, shape, device)
    }
}

#[derive(Copy, Clone)]
pub struct LogNormalWeightDistribution {
    pub loc: f32,
    pub scale: f32
}

impl LogNormalWeightDistribution {
    pub fn new(loc: f32, scale: f32) -> LogNormalWeightDistribution {
        LogNormalWeightDistribution {
            loc,
            scale
        }
    }
}

impl WeightDistribution<(usize, usize)> for LogNormalWeightDistribution {
    fn sample(&mut self, shape: (usize, usize), device: &Device) -> candle_core::Result<Tensor> {
        Tensor::randn(0.0, 1.0, shape, device)?
            .affine(self.scale as f64, self.loc as f64)?
            .exp()
    }
}