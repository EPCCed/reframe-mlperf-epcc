import reframe as rfm
import reframe.utility.sanity as sn

class CosmoFlowBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()

    @performance_function("images/s", perf_key="Throughput")
    def extract_throughput(self):
        return sn.extractsingle(r"Processing Speed: (.*)", self.stdout, tag= 1, conv=float)
    
    @performance_function("s", perf_key="Communication Time")
    def extract_communication(self):
        return sn.extractsingle(r"Communication Time: (.*)", self.stdout, tag= 1, conv=float)

    @sanity_function
    def assert_target_met(self):
        return sn.assert_found(r'Processing Speed', filename=self.stdout)