mod absolute;
mod joystick;
mod predictor;
mod relative;

pub use absolute::AbsolutePointerProducer;
pub use joystick::JoystickPointerProducer;
pub use predictor::{
    DecayingVelocityPointerPredictor, DecayingVelocityPointerPredictorConfig,
    KinematicPointerPredictor, KinematicPointerPredictorConfig, OneEuroVelocityPointerPredictor,
    OneEuroVelocityPointerPredictorConfig, VelocityPointerPredictor,
};
pub use relative::RelativePointerProducer;
