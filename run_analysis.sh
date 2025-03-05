#!/bin/bash

# Set up error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Create log directory
mkdir -p logs

# Create necessary directories
mkdir -p plots/sensitivity
mkdir -p plots/stability/vehicle
mkdir -p plots/stability/transaction
mkdir -p plots/calibration
mkdir -p plots/model_stability
mkdir -p plots/cross_validation
mkdir -p plots/robustness
mkdir -p plots/learning_curve
mkdir -p plots/test_evaluation
mkdir -p plots/hyperparameter_tuning
mkdir -p plots/model_comparison
mkdir -p plots/eda
mkdir -p plots/heuristics
mkdir -p plots/clustering
mkdir -p plots/modelling
mkdir -p results/flag_validation

# Function to run a python script and log output
run_script() {
    script_path=$1
    script_name=$(basename "$script_path")
    log_file="logs/${script_name%.py}.log"
    
    echo "====================================================="
    echo "Running: $script_name"
    echo "====================================================="
    
    # Run the script and capture output
    python "$script_path" 2>&1 | tee "$log_file"
    
    # Check if script completed successfully
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Completed: $script_name"
    else
        echo "❌ Failed: $script_name"
        exit 1
    fi
}

echo "=== Starting Fuel Analysis Pipeline ==="

# Step 1: Original data preparation and analysis scripts - we do not run data prep since it has already been done
echo "=== STEP 1: Original Data Preparation and Analysis ==="
run_script "code-only/Final_EDA.py"
run_script "code-only/Final_Clustering.py" 
run_script "code-only/Heuristics.py"
run_script "code-only/Predictive Models.py"

# Step 2: New sensitivity and validation analysis scripts
echo "=== STEP 2: Sensitivity and Validation Analyses ==="
run_script "code-only/heuristic-sensitivity-analysis.py"
run_script "code-only/clustering-stability-analysis.py"
run_script "code-only/classification-validation.py"

echo "=== Analysis Pipeline Complete ==="