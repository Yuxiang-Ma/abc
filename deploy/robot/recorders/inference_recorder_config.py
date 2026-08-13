from dataclasses import dataclass


@dataclass
class InferenceRecorderConfig:
    name: str = "InferenceRecorder"
    control_rate: float = 1000.0
    data_root_directory: str = ""
    collection_name: str = ""

    # Policy metadata — written as H5 root attributes
    checkpoint_path: str = ""
    model_size: str = ""
    diffusion_steps: int = 10

    rtc: bool = False
    rtc_prefix_length: int = 0
    rtc_inference_lead_steps: int = 0

    task_name: str = ""
    """Written as H5 root attribute."""
    session_tag: str = ""
    dagger: bool = False
