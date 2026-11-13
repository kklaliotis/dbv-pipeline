import os
import numpy as np
from .base_stage import PipelineStage
from .data_types import YamlFile, Directory # KL We need to add JSONFile

class MeasurePSF(PipelineStage):
    """
    This stage measures the PSF on the cutout image and saves some stats to a JSON file. 

    Inputs:
    Outputs:
    To-do:
        - Implement the actual PSF measurement functionality
        -
    """

    name = "MeasurePSF"
    inputs = [("isolated_cutouts", Directory)]
    outputs = [("psf_stats", YamlFile)]
    config_options = {}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)

        isolated_cutout_directory = self.get_input("isolated_cutouts")
        
        measure_psfs(my_config, isolated_cutout_directory)  # Placeholder 
        compute_kernels(my_config, isolated_cutout_directory)  # Placeholder 


        psf_stats = self.get_output("psf_stats")
        print(f"PSF Stage wrote PSF stats to {psf_stats}")

