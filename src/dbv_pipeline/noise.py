import os
import numpy as np
from .base_stage import PipelineStage
from .data_types import YamlFile, Directory # KL We need to add JSONFile

class MeasureNoise(PipelineStage):
    """
    This stage measures the noise in empty regions of the images and outputs noise statistics and noise maps.

    Inputs:
    Outputs:
    To-do:
        - Implement the actual noise measurement functionality
        - Add noise QA visualization
    """

    name = "MeasureNoise"
    inputs = [("empty_cutouts", Directory)]
    outputs = [("noise_model", YamlFile), ("noise_qa", Directory)]
    config_options = {}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

        empty_cutout_dir = self.get_input("empty_cutouts")
        
        measure_noise(my_config, empty_cutout_dir)  # Placeholder 

        noise_model = self.get_output("noise_model")
        print(f"Noise Stage wrote noise model to {noise_model}")
        noise_qa = self.get_output("noise_qa")
        print(f"Noise Stage wrote noise QA to {noise_qa}")

