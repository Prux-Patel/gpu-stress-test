# GPU Benchmark Script with LLM Tests and Logging

This repository contains a **Bash script** for automating GPU benchmarking using Large Language Models (LLMs).  
The script installs dependencies, verifies GPU + software setup, runs inference benchmarks on multiple models (Mistral, LLaMA 3, Gemma 3), collects GPU metrics, and automatically pushes results to GitHub.

---

## ✨ Features

- 🔧 **Dependency management** – Installs required packages (PyTorch, CUDA, Ollama, etc.)
- ✅ **Verification checks** – Confirms GPU and tools are correctly installed
- ⚡ **GPU benchmarking** – Runs inference tests with:
  - [Mistral](https://mistral.ai/)
  - [LLaMA 3](https://ai.meta.com/llama/)
  - [Gemma 3](https://ai.google/gemma/)
- 📊 **Metric collection** – Logs:
  - GPU utilization (%)
  - Memory usage (MB)
  - Temperature (°C)
  - Inference throughput (inferences/sec)
- 📝 **Reports & logs** – Saves CSV metrics and summary reports
- 🚀 **GitHub integration** – Pushes benchmark files into a results repository

---

## 📂 Repository Structure

```
gpu-benchmark.sh                  # Main script
collate_and_import_to_sqlite.sh   # Optional CSV → SQLite utility
README.md                         # Documentation
```

Generated files after running benchmarks:
```
gpu_test_log.txt          # Full log with timestamps
benchmark_report.txt      # Final benchmark summary
*_benchmark.csv           # Per-model CSV results
```

---

## ⚙️ Prerequisites

- Ubuntu 22.04+ (recommended)
- NVIDIA GPU with CUDA support
- GitHub account with access to push results
- Internet access (for installing packages and pulling models)

---

## 🚀 Installation & Usage

1. **Clone this repository**
   ```bash
   git clone https://github.com/Prux-Patel/gpu-benchmark-results.git
   cd gpu-benchmark-results
   ```

2. **Make the script executable**
   ```bash
   chmod +x gpu-benchmark.sh
   ```

3. **Run the script**
   ```bash
   ./gpu-benchmark.sh
   ```

4. **Follow the interactive menu**:
   ```
   === GPU Benchmark Menu ===
   1. Install and Verify Dependencies
   2. Run Test on Mistral
   3. Run Test on LLaMA 3
   4. Run Test on Gemma 3
   5. Push Benchmark Files to GitHub
   6. Run All (1 to 5 + verify install)
   7. Collate CSVs and Import to SQLite
   8. Exit
   ```

---

## 📊 Example Output

### **CSV file (mistral_benchmark.csv)**
```csv
Timestamp,Inference Count,GPU Memory (MB),GPU Utilization (%),Temperature (C),Throughput (inferences/sec)
12:00:00,10,3400,72,65,0.50
12:02:00,20,3450,75,66,0.55
```

### **Report file (benchmark_report.txt)**
```
===================================
      GPU Benchmark Report        
===================================
Model Tested: Mistral
Total Inferences: 200
Total Duration: 600 seconds
Average GPU Memory Usage: 3450 MB
Average GPU Utilization: 74%
Average GPU Temperature: 66°C
Final Throughput: 0.55 inferences/sec
===================================
```

---

## 🔄 GitHub Integration

The script clones/pulls the results repository, copies generated logs/CSV files, and pushes them to the specified branch (`main` by default).

You may configure in `gpu-benchmark.sh`:
```bash
REPO_URL="https://github.com/Prux-Patel/gpu-benchmark-results.git"
BRANCH_NAME="main"
```

---

## 🛠️ Optional: SQLite Integration

If you want to collate results and load them into a SQLite database for visualization (e.g., Grafana):

```bash
bash collate_and_import_to_sqlite.sh
```

---

## 🤝 Contributing

1. Fork this repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to your branch (`git push origin feature-name`)  
5. Open a Pull Request  

---

## 📜 License

MIT License – feel free to use, modify, and share.

---

## 👤 Author

**Prakash Patel**  
🔗 [GitHub: Prux-Patel](https://github.com/Prux-Patel)
