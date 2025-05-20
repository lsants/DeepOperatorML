from .base_training_strategy import TrainingStrategy
from ..components.branch import branch_factory
from ..components.trunk import trunk_factory
class PODStrategy(TrainingStrategy):
    def initialize_components(self):
        self.branch = BranchRegistry.create(
            self.train_cfg.BRANCH.ARCHITECTURE,
            self.train_cfg.BRANCH
        )
        self.trunk = TrunkRegistry.create(
            'pod',  # Force POD trunk
            self.train_cfg.TRUNK
        )
    
    def get_phases(self):
        return [PhaseConfig('pod', self.train_cfg.STRATEGY_PARAMS.pod.epochs, ['branch'])]
    
    def preprocess_data(self, data):
        self.trunk.compute_basis(data, self.train_cfg.STRATEGY_PARAMS.pod.var_share)
    
    def configure_output_handling(self):
        # POD requires fixed output dimensions
        self.output_handler = FixedBasisHandler(self.trunk.basis)