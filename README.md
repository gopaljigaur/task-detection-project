# Task Detection Project

## Project Setup Instructions

These instructions will guide you through the setup process for the project. Make sure you have [conda](https://docs.conda.io/en/latest/) installed before proceeding.

### Step 1: Clone the Repository

Clone the project repository to your local machine using the following command:

`git clone https://github.com/gopaljigaur/task-detection-project`

### Step 2: Create and Activate the Conda Environment

Create a new conda environment using the following command:

`conda create --name <environment_name>`

And then activate the environment using:

`conda activate <environment_name>`

### Step 3: Install Project Dependencies

Use `make` to install the project dependencies and set up the necessary configurations. Choose the appropriate command based on your requirements:

- For the full interactive setup including dependencies for Vision Transformer and maniskill2 setup, run:

  `make setup`

- For a full non-interactive setup, run:

  `make setup-ni`

- To setup only dino-vit dependencies, run:

  `make setup-vit`

- To setup only the maniskill2 simulation environment and download assets, run:

  `make setup-mani`

### Step 4: Start Using the Project

At this point, the project is set up and ready to use. You can now proceed with your desired workflow.


#### Resources
[dino-vit-features](https://github.com/ShirAmir/dino-vit-features), [maniskill2](https://maniskill2.github.io/)