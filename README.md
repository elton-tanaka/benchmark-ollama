# Benchmark-Ollama

## 📌 Overview
This project benchmarks CPU, RAM, and GPU usage while running AI models using **Ollama**. It automatically logs system resource usage and saves a performance report as an image.

---

## ⚙️ Installation Guide

### **🖥️ Windows Installation**
1. **Install Ollama**
   - Download and install Ollama from the official site:
     ```sh
     curl -fsSL https://ollama.com/install.sh | sh
     ```

2. **Install Python & Create a Virtual Environment**
   - Ensure **Python 3.8+** is installed. If not, download it from [python.org](https://www.python.org/downloads/).
   - Create a virtual environment and activate it:
     ```sh
     python -m venv venv
     source venv/bin/activate  # On Linux/Mac
     venv\Scripts\activate  # On Windows
     ```
   - Install dependencies:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run the Benchmark**
   - Clone or download this repository:
     ```sh
     git clone https://github.com/your-repo/benchmark-ollama.git
     cd benchmark-ollama
     ```
   - Run the script:
     ```sh
     python benchmark.py
     ```

4. **View the Results**
   - The benchmark results will be saved as an image in the project folder, with a timestamp.
   - Example: `20240219_153000_results.png`

---

### **🐧 Linux Installation**

1. **Install Ollama**
   ```sh
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Install Python & Create a Virtual Environment**
   - Ensure **Python 3.8+** is installed:
     ```sh
     sudo apt update && sudo apt install python3 python3-pip python3-venv
     ```
   - Create a virtual environment and activate it:
     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Install dependencies:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run the Benchmark**
   ```sh
   git clone https://github.com/your-repo/benchmark-ollama.git
   cd benchmark-ollama
   python benchmark.py
   ```

4. **Check the Results**
   - The performance report will be saved as a PNG file in the same directory.

---

## 🔥 Features
- ✅ **Benchmarks CPU, RAM, and GPU usage**
- ✅ **Supports NVIDIA GPUs (automatically detects if available)**
- ✅ **Generates a performance report image with system usage stats**
- ✅ **Works on Windows & Linux**
- ✅ **Uses a virtual environment for cleaner dependency management**

## ⚠️ Notes
- AMD GPUs are **not currently supported**. If you need AMD support, let us know!
- The benchmark only runs **Ollama-compatible models**.

## 📜 License
This project is **open-source**. Feel free to modify and contribute!

🚀 Happy benchmarking!

