import reframe.utility.sanity as sn
from ..gpu import ResNet50GPUBenchmark

class CPUFullTest(ResNet50GPUBenchmark):
    def __init__(self):
        super().__init__()
        
    @sanity_function
    def assert_target_met(self):
        return sn.assert_found(r'"key": "run_stop"', filename=self.stdout)