"""Configuration for DDIM sampling."""

from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for DDIM sampling."""
    
    num_inference_steps: int = 50
    schedule_type: str = "uniform"  # "uniform", "quadratic", or "custom"
    eta: float = 0.0  # Stochasticity parameter (0 = deterministic)
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"  # "linear", "scaled_linear", "squaredcos_cap_v2"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}"
            )
        
        if self.num_inference_steps > 1000:
            raise ValueError(
                f"num_inference_steps must be <= 1000, got {self.num_inference_steps}"
            )
        
        valid_schedules = ["uniform", "quadratic", "custom"]
        if self.schedule_type not in valid_schedules:
            raise ValueError(
                f"Invalid schedule_type '{self.schedule_type}'. "
                f"Valid options are: {valid_schedules}"
            )
        
        if not 0 <= self.eta <= 1:
            raise ValueError(
                f"eta must be in range [0, 1], got {self.eta}"
            )
        
        valid_beta_schedules = ["linear", "scaled_linear", "squaredcos_cap_v2"]
        if self.beta_schedule not in valid_beta_schedules:
            raise ValueError(
                f"Invalid beta_schedule '{self.beta_schedule}'. "
                f"Valid options are: {valid_beta_schedules}"
            )
