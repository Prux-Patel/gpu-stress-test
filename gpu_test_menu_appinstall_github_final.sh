#!/bin/bash

# ================================================
# GPU Benchmark Script with LLM Tests and Logging
# ================================================
# This script installs dependencies, verifies installations,
# runs performance benchmarks for several LLMs (Mistral, LLaMA 3, Gemma 3),
# captures GPU metrics, and pushes the results to GitHub.

# --------------------------------------------
# Bash safety flags to catch script failures
# --------------------------------------------
# -e  : Exit script if a command fails
# -u  : Treat unset variables as errors
# -o pipefail : Catch errors in piped commands
set -euo pipefail

# -------------------------
# File paths for logs/reports
# -------------------------
LOG_FILE="gpu_test_log.txt"
REPORT_FILE="benchmark_report.txt"

# -------------------------
# GitHub configuration for pushing results
# -------------------------
REPO_URL="https://github.com/Prux-Patel/gpu-benchmark-results.git"
REPO_DIR="/tmp/gpu-benchmark-results"
BRANCH_NAME="main"

# --------------------------------------------
# Utility Function: Sanitize input
# --------------------------------------------
# Removes unsafe characters to protect against command injection
sanitize_input() {
  local input="$1"
  printf '%s\n' "$input" | sed 's/[^a-zA-Z0-9@._-]//g'
}

# --------------------------------------------
# Logging Function
# --------------------------------------------
# Adds timestamp to log messages and writes to file and console
log() {
  printf "%s - %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a "$LOG_FILE"
}

# --------------------------------------------
# Install all required packages and models
# --------------------------------------------
install_dependencies() {
  log "Starting installation of dependencies..."

  # Update package list and upgrade system packages
  sudo apt update && sudo apt upgrade -y

  # Install system packages for GPU tests and Python tools
  sudo apt install -y python3-pip nvidia-cuda-toolkit git bc

  # Install PyTorch with CUDA 12.1 support for GPU acceleration
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  # Install Ollama - used for running LLMs locally
  curl -fsSL https://ollama.com/install.sh | sh

  # Install yq - YAML processor, useful for config file operations
  snap install yq

  # Pull LLM models to be used in benchmarks
  ollama pull mistral
  ollama pull llama3
  ollama pull gemma3

  log "Installation completed."
}

# --------------------------------------------
# Verify software installations
# --------------------------------------------
# Confirms key tools are available and functional
verify_installation() {
  log "Verifying installed applications..."

  local -a checks=(
    "python3 --version"
    "pip --version"
    "nvidia-smi"
    "git --version"
    "bc --version"
    "python3 -c 'import torch; print(torch.cuda.is_available())'"
    "ollama --version"
    "yq --version"
  )

  # Execute each check and log result
  for check in "${checks[@]}"; do
    if eval "$check" > /dev/null 2>&1; then
      log "[✔] $check succeeded"
    else
      log "[✘] $check failed"
    fi
  done

  log "Verification completed."
}

# --------------------------------------------
# Run GPU inference test for a specific LLM
# --------------------------------------------
# model: LLM model to test
# prompt: Text to pass into the model
run_llm_test() {
  local model="$1"
  local prompt="$2"
  local csv_file="${model}_benchmark.csv"

  log "Starting $model test..."

  local log_interval=120       # Collect GPU data every 2 minutes
  local total_duration=600     # Test duration: 10 minutes
  local num_parallel=2         # Number of parallel inferences per cycle
  local start_time; start_time=$(date +%s)
  local end_time=$((start_time + total_duration))

  # CSV header for tracking GPU stats
  printf "Timestamp,Inference Count,GPU Memory (MB),GPU Utilization (%%),Temperature (C),Throughput (inferences/sec)\n" > "$csv_file"

  # Initialize accumulators for averages
  local inference_count=0
  local memory_sum=0
  local utilization_sum=0
  local temperature_sum=0
  local log_count=0

  # Loop until test time is completed
  while [[ $(date +%s) -lt "$end_time" ]]; do
    local current_time; current_time=$(date +%s)

    # Launch concurrent inferences
    for ((i=1; i<=num_parallel; i++)); do
      ollama run "$model" "$prompt" > /dev/null &
    done
    wait

    # Update inference count
    inference_count=$((inference_count + num_parallel))

    # Collect current GPU metrics
    local memory_usage; memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
    local gpu_utilization; gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{print $1}')
    local gpu_temp; gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | awk '{print $1}')

    # Add to running totals for averages
    memory_sum=$((memory_sum + memory_usage))
    utilization_sum=$((utilization_sum + gpu_utilization))
    temperature_sum=$((temperature_sum + gpu_temp))
    log_count=$((log_count + 1))

    # Calculate throughput since start
    local elapsed=$((current_time - start_time))
    local throughput
    if [[ "$elapsed" -gt 0 ]]; then
      throughput=$(echo "scale=2; $inference_count / $elapsed" | bc)
    else
      throughput=0
    fi

    # Save metrics to CSV
    printf "%s,%s,%s,%s,%s,%s\n" "$(date '+%H:%M:%S')" "$inference_count" "$memory_usage" "$gpu_utilization" "$gpu_temp" "$throughput" >> "$csv_file"

    # Display progress on terminal
    local percentage=$((elapsed * 100 / total_duration))
    printf "\r⏳ %s%% | Inferences: %s | GPU: %s%% | Mem: %s MB | Temp: %s°C | Throughput: %s inferences/sec    " "$percentage" "$inference_count" "$gpu_utilization" "$memory_usage" "$gpu_temp" "$throughput"

    # Log the stats
    log "Inference #$inference_count - GPU Memory: $memory_usage MB - Utilization: $gpu_utilization%% - Temp: $gpu_temp°C - Throughput: $throughput inferences/sec"

    # Wait until next log interval
    sleep "$log_interval"
  done

  # Final metrics
  local duration=$((current_time - start_time))
  local final_throughput=$(echo "scale=2; $inference_count / $duration" | bc)
  local avg_memory=$(echo "scale=2; $memory_sum / $log_count" | bc)
  local avg_utilization=$(echo "scale=2; $utilization_sum / $log_count" | bc)
  local avg_temp=$(echo "scale=2; $temperature_sum / $log_count" | bc)

  # Final logs and summary
  log "$model Test Completed"
  log "Total Inferences: $inference_count"
  log "Total Duration: $duration seconds"
  log "Average GPU Memory Usage: $avg_memory MB"
  log "Average GPU Utilization: $avg_utilization%%"
  log "Average GPU Temperature: $avg_temp°C"
  log "Final Throughput: $final_throughput inferences/sec"

  # Append a formatted summary to report file
  {
    printf "===================================\n"
    printf "      GPU Benchmark Report        \n"
    printf "===================================\n"
    printf "Model Tested: %s\n" "$model"
    printf "Total Inferences: %s\n" "$inference_count"
    printf "Total Duration: %s seconds\n" "$duration"
    printf "Average GPU Memory Usage: %s MB\n" "$avg_memory"
    printf "Average GPU Utilization: %s%%\n" "$avg_utilization"
    printf "Average GPU Temperature: %s°C\n" "$avg_temp"
    printf "Final Throughput: %s inferences/sec\n" "$final_throughput"
    printf "===================================\n\n"
  } >> "$REPORT_FILE"
}

# --------------------------------------------
# Push results to GitHub repository
# --------------------------------------------
push_to_github() {
  log "Pushing results to GitHub..."

  # Remove previous clone
  if [[ -d "$REPO_DIR" ]]; then
    rm -rf "$REPO_DIR"
  fi

  # Clone the repo fresh
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR" || return 1

  # Switch to or create the target branch
  git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME"

  # Copy new files into repo folder
  cp -f "$OLDPWD/$LOG_FILE" "$OLDPWD/$REPORT_FILE" "$OLDPWD/"*_benchmark.csv "$REPO_DIR"/

  # Stage, commit, and push
  git add *.txt *.csv
  git config user.name "gpu-benchmark-bot"
  git config user.email "noreply@benchmark.local"
  git commit -m "Automated GPU benchmark results update: $(date '+%Y-%m-%d %H:%M:%S')"
  git push origin "$BRANCH_NAME"

  log "Push to GitHub completed successfully."
}

# --------------------------------------------
# Run all major steps sequentially
# --------------------------------------------
run_all() {
  install_dependencies
  verify_installation
  run_llm_test "mistral" "Summarize the impact of AI in healthcare."
  run_llm_test "llama3" "Describe the history of neural networks."
  run_llm_test "gemma3" "Discuss the ethical implications of AI in business."
  push_to_github
}

# --------------------------------------------
# Display an interactive menu to users
# --------------------------------------------
main() {
  while true; do
    printf "\n=== GPU Benchmark Menu ===\n"
    printf "1. Install and Verify Dependencies\n"
    printf "2. Run Test on Mistral\n"
    printf "3. Run Test on LLaMA 3\n"
    printf "4. Run Test on Gemma 3\n"
    printf "5. Push Benchmark Files to GitHub\n"
    printf "6. Run All (1 to 5 + verify install)\n"
    printf "7. Collate CSVs and Import to SQLite\n"
    printf "8. Exit\n"

    printf "Select an option: "
    read -r choice

    case "$choice" in
      1) install_dependencies; verify_installation ;;
      2) run_llm_test "mistral" "Summarize the impact of AI in healthcare." ;;
      3) run_llm_test "llama3" "Describe the history of neural networks." ;;
      4) run_llm_test "gemma3" "Discuss the ethical implications of AI in business." ;;
      5) push_to_github ;;
      6) run_all ;;
      7) bash collate_and_import_to_sqlite.sh ;;
      8) printf "Goodbye!\n"; break ;;
      *) printf "Invalid option. Please select 1-8.\n" >&2 ;;

      *) printf "Invalid option. Please select 1-7.\n" >&2 ;;
    esac
  done
}

main
