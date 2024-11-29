import wandb
import yaml
from main import main

# Load the configuration from the YAML file
with open("../run_config.yaml") as file:
    config = yaml.safe_load(file)

# Initialize Weights & Biases with the loaded configuration
wandb.init(project="RL-Final-Project", config=config)

# Run the main training function
main()
