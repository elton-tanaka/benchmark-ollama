# Benchmark-Ollama

## 📌 Overview
This project benchmarks CPU, RAM, and GPU usage while running AI models using **Ollama**. It automatically logs system resource usage and saves a performance report as an image.

---

## ⚙️ Installation Guide

### **🖥️ Windows Installation**
1. **Install Ollama**
   - Download and install Ollama from the official site:
     https://ollama.com/download/windows

2. **Download a Model in Ollama**
   - Before running the benchmark, you must download an AI model compatible with Ollama:
     ```sh
     ollama pull <model-name>
     ```
   - Replace `<model-name>` with an available model from [Ollama's official models](https://ollama.com/library/).
   - Example:
     ```sh
     ollama pull deepseek-r1:14b
     ```

3. **Install Python & Create a Virtual Environment**
   - Ensure **Python 3.8+** is installed. If not, download it from [python.org](https://www.python.org/downloads/).
   - Create a virtual environment and activate it:
     ```sh
     python3 -m venv env
     source env/bin/activate  # On Linux/Mac
     env\Scripts\activate.bat  # On Windows
     ```
   - Install dependencies from `requirements.txt`:
     ```sh
     pip install -r requirements.txt
     ```

4. **Run the Benchmark**
   - Clone or download this repository:
     ```sh
     git clone https://github.com/your-repo/benchmark-ollama.git
     cd benchmark-ollama
     ```
   - Edit `benchmark.py` to change the model name:
     ```python
     model_name = "mistral"  # Change to your preferred model
     ```
   - Run the script:
     ```sh
     python benchmark.py
     ```

5. **View the Results**
   - The benchmark results will be saved as an image in the project folder, with a timestamp.
   - Example: `20240219_153000_results.png`

---

### **🐧 Linux Installation**

1. **Install Ollama**
   ```sh
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Download a Model in Ollama**
   - Before running the benchmark, download an AI model:
     ```sh
     ollama pull <model-name>
     ```
   - Example:
     ```sh
     ollama pull mistral
     ```

3. **Install Python & Create a Virtual Environment**
   - Ensure **Python 3.8+** is installed:
     ```sh
     sudo apt update && sudo apt install python3 python3-pip python3-venv
     ```
   - Create a virtual environment and activate it:
     ```sh
     python3 -m venv env
     source env/bin/activate
     ```
   - Install dependencies from `requirements.txt`:
     ```sh
     pip install -r requirements.txt
     ```

4. **Run the Benchmark**
   ```sh
   git clone https://github.com/your-repo/benchmark-ollama.git
   cd benchmark-ollama
   ```
   - Edit `benchmark.py` to change the model name:
     ```python
     model_name = "mistral"  # Change to your preferred model
     ```
   - Run the script:
     ```sh
     python benchmark.py
     ```

5. **Check the Results**
   - The performance report will be saved as a PNG file in the same directory.

---

## 🔥 Features
- ✅ **Benchmarks CPU, RAM, and GPU usage**
- ✅ **Supports NVIDIA GPUs (automatically detects if available)**
- ✅ **Generates a performance report image with system usage stats**
- ✅ **Works on Windows & Linux**
- ✅ **Uses a virtual environment for cleaner dependency management**
- ✅ **Supports custom AI models from Ollama**

## ⚠️ Notes
- **You must download a model before running the benchmark.**
- AMD GPUs are **not currently supported**. If you need AMD support, let us know!
- The benchmark only runs **Ollama-compatible models**.

## 📜 Managing Dependencies
- If you install new packages, update `requirements.txt` by running:
  ```sh
  pip freeze > requirements.txt
  ```
- To install dependencies from `requirements.txt`, use:
  ```sh
  pip install -r requirements.txt
  ```

## 📜 License
This project is **open-source**. Feel free to modify and contribute!

🚀 Happy benchmarking!

