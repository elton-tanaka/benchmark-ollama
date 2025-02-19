import ollama
import psutil
import time
import threading
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Try to import NVIDIA GPU monitoring library
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle).decode()  # Get GPU model
except:
    gpu_available = False
    gpu_name = None

# Get CPU & RAM Info
cpu_name = f"{psutil.cpu_freq().max:.0f} MHz" if psutil.cpu_freq() else "Unknown CPU"
ram_total = round(psutil.virtual_memory().total / (1024**3), 2)  # Convert bytes to GB

# Global variables to store real-time stats
cpu_usage = []
ram_usage = []
gpu_usage = []  # GPU usage tracking (only if available)
timestamps = []
start_time = None
stop_event = threading.Event()
inference_time = 0  # Store inference time

# Function to track system usage
def track_system_usage():
    global start_time
    start_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time() - start_time
        timestamps.append(current_time)
        cpu_usage.append(psutil.cpu_percent(interval=0.5))
        ram_usage.append(psutil.virtual_memory().percent)
        
        # If GPU is available, track GPU usage
        if gpu_available:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            gpu_usage.append(gpu_util)

# Function to generate and save the final benchmark plot
def save_final_plot():
    plt.figure(figsize=(8, 8))  # Increased plot size

    # Calculate max and average values dynamically
    max_cpu = max(cpu_usage) if cpu_usage else 0
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    max_ram = max(ram_usage) if ram_usage else 0
    avg_ram = sum(ram_usage) / len(ram_usage) if ram_usage else 0

    # Set title with hardware info
    title_text = f"CPU: {cpu_name} | RAM: {ram_total} GB"
    if gpu_available:
        title_text += f" | GPU: {gpu_name}"
    title_text += f"\nInference Time: {inference_time:.2f}s"
    
    plt.suptitle(title_text, fontsize=12, fontweight="bold")

    # Plot CPU Usage
    ax1 = plt.subplot(3 if gpu_available else 2, 1, 1)  # 3 graphs if GPU is available
    ax1.plot(timestamps, cpu_usage, label="CPU Usage (%)", color="red")
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("CPU Usage (%)")
    ax1.legend()
    ax1.text(0.75 * max(timestamps, default=1), 90, f"Max CPU: {max_cpu:.2f}%", fontsize=10, color="red")
    ax1.text(0.75 * max(timestamps, default=1), 80, f"Avg CPU: {avg_cpu:.2f}%", fontsize=10, color="red")

    # Plot RAM Usage
    ax2 = plt.subplot(3 if gpu_available else 2, 1, 2)
    ax2.plot(timestamps, ram_usage, label="RAM Usage (%)", color="blue")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RAM Usage (%)")
    ax2.legend()
    ax2.text(0.75 * max(timestamps, default=1), 90, f"Max RAM: {max_ram:.2f}%", fontsize=10, color="blue")
    ax2.text(0.75 * max(timestamps, default=1), 80, f"Avg RAM: {avg_ram:.2f}%", fontsize=10, color="blue")

    # If GPU is available, plot GPU Usage
    if gpu_available:
        max_gpu = max(gpu_usage) if gpu_usage else 0
        avg_gpu = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(timestamps, gpu_usage, label="GPU Usage (%)", color="green")
        ax3.set_ylim(0, 100)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("GPU Usage (%)")
        ax3.legend()
        ax3.text(0.75 * max(timestamps, default=1), 90, f"Max GPU: {max_gpu:.2f}%", fontsize=10, color="green")
        ax3.text(0.75 * max(timestamps, default=1), 80, f"Avg GPU: {avg_gpu:.2f}%", fontsize=10, color="green")

    # Save the plot as an image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    save_path = os.path.abspath(f"{timestamp}_results.png")  # Filename with timestamp
    plt.savefig(save_path, bbox_inches='tight')  # Save as PNG
    plt.close()  # Close the plot after saving
    print(f"📸 Saved benchmark results at: {save_path}")

# Function to benchmark Ollama AI model
def benchmark_ollama(model: str, prompt: str):
    global inference_time

    print(f"Starting benchmark for model: {model}")

    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=track_system_usage)
    monitor_thread.start()

    # Run Ollama model
    start_time = time.time()
    ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    inference_time = end_time - start_time  # Store inference time

    # Stop monitoring
    stop_event.set()
    monitor_thread.join()

    # Display benchmark summary
    print(f"\n📊 Benchmark Results for {model}:")
    print(f"⏳ Inference Time: {inference_time:.2f} seconds")
    
    if gpu_available:
        print(f"🎮 GPU Detected: {gpu_name}")

    # Save the final plot
    save_final_plot()

# Example Usage
model_name = "deepseek-r1:1.5b"  # Change to your Ollama model
test_prompt = "Explain machine learning in simple terms."

# Run the benchmark
benchmark_ollama(model_name, test_prompt)
