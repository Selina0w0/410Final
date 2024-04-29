#!/bin/bash
#SBATCH --job-name=spark_job          # Job name
#SBATCH --nodes=4                     # Number of nodes to request
#SBATCH --ntasks-per-node=4           # Number of processes per node
#SBATCH --mem=8G                      # Memory per node
#SBATCH --time=1:30:00                # Maximum runtime in HH:MM:SS
#SBATCH --account=open 	      # Queue
#SBATCH --mail-user=skf5373@psu.edu
#SBATCH --mail-type=BEGIN
# Load necessary modules (if required)
module load anaconda3
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0

# Create a new conda environment
conda create -y --name skf5373_final python=3.8
source activate skf5373_final

export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

# Download and install sparktorch
pip install git+https://github.com/dmmiller612/sparktorch.git
pip install examples
wget https://raw.githubusercontent.com/dmmiller612/sparktorch/master/examples/mnist_train.csv -O mnist_train.csv

# Run PySpark
# Record the start time
start_time=$(date +%s)
spark-submit --deploy-mode client skf5373.py

#python skf5373.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"

conda deactivate