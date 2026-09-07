#[derive(Copy, Clone, Debug)]
pub struct NeuronParameters {
    pub u_t: f32,
    pub u_rest: f32,

    pub c_mem: f32,
    pub g_leak: f32,

    pub theta_t: f32,
    pub theta_rest: f32,

    pub tau_theta: f32,

    pub theta_reset: f32,

    pub g_exc: f32,
    pub e_exc: f32,

    pub tau_exc: f32,

    pub g_inh: f32,
    pub e_inh: f32,

    pub tau_inh: f32
}

impl Default for NeuronParameters {
    fn default() -> Self {
        NeuronParameters {
            u_t: -65.0,
            u_rest: -65.0,

            c_mem: 200.0,
            g_leak: 10.0,

            theta_t: -55.0,
            theta_rest: -55.0,

            tau_theta: 200.0,

            theta_reset: -75.0,

            g_exc: 0.0,
            e_exc: 0.0,

            tau_exc: 5.0,

            g_inh: 0.0,
            e_inh: -80.0,

            tau_inh: 10.0
        }
    }
}