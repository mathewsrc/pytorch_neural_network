from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

# Define model hyperparameters
hyperparameters = {"epochs": "2", "batch-size": "64", "test-batch-size": "100", "lr": "0.001"}

# Create a Pytorch estimator
estimator = Pytorch(
    entry_point="pytorch_cifar.py",
    role=get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.large",
    hyperparameters=hyperparameters,
    framework_version="1.8",
    py_version="py36"
)

# Train model
estimator.fit(wait=True)