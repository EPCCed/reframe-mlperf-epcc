import reframe.utility.sanity as sn
from ..cpu import ResNet50CPUCheck

class CPUFullTest(ResNet50CPUCheck):
    def __init__(self):
        super().__init__()
    
        

    @sanity_function
    def assert_target_met(self):
        return sn.assert_found(r'"key": "run_stop"', filename=self.stdout)