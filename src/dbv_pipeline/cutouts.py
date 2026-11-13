import os
import numpy as np
from .base_stage import PipelineStage
from .data_types import ParquetFile, Directory # KL We need to add JSONFile

class MakeCutouts(PipelineStage):
    """
    This pipeline element is for creating cutouts of galaxies (from the catalog) from the input images.
    Inputs:
    Outputs:
    To-do:
        - Implement the actual cutout functionality
        - Not sure the right way to access images - could direct this to a directory with the info,
            but I expect we will be streaming the data from wherever so I think this will maybe be a URL
            or something like that... TBD
        - Similar for outputs; don't know the exact format but I think it will be a directory with the
            cutouts in it, 
    """

    name = "MakeCutouts"
    inputs = [("selection_catalog", ParquetFile)]
    outputs = [("isolated_cutouts", Directory), ("empty_cutouts", Directory)]
    config_options = {}

    def run(self):
        # Retrieve configuration:
        my_config = self.config
        print("Here is my configuration :", my_config)
        
        make_cutouts(my_config)  # Placeholder for the actual cutout functionality

        isolated_cutout_directory = self.get_output("isolated_cutouts")
        print(f"Cutouts Stage wrote isolated cutout images to {isolated_cutout_directory}")

        empty_cutout_directory = self.get_output("empty_cutouts")
        print(f"Cutouts Stage wrote empty cutout images to {empty_cutout_directory}")

