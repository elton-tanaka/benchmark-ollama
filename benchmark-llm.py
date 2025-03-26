import ollama
import psutil
import time
import threading
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from nltk.translate.bleu_score import sentence_bleu
import math

# Define the models to benchmark (Compressed vs. Non-Compressed)
MODELS = [
    "mistral",  # Example of a standard model
    "mistral:8bit",  # Example of a compressed model
]

# Get system hardware info
cpu_name = f"{psutil.cpu_freq().max:.0f} MHz" if psutil.cpu_freq() else "Unknown CPU"
ram_total = round(psutil.virtual_memory().total / (1024**3), 2)  # Convert bytes to GB

timestamps = []
cpu_usage = {model: [] for model in MODELS}
ram_usage = {model: [] for model in MODELS}
inference_times = {}
bleu_scores = {}
perplexity_scores = {}
stop_event = threading.Event()

REFERENCE_RESPONSE = ["Artificial intelligence is the simulation of human intelligence by machines."]

def track_system_usage(model):
    while not stop_event.is_set():
        timestamps.append(time.time())
        cpu_usage[model].append(psutil.cpu_percent(interval=0.5))
        ram_usage[model].append(psutil.virtual_memory().percent)

def calculate_perplexity(response_text):
    words = response_text.split()
    N = len(words)
    if N == 0:
        return float('inf')  # Avoid division by zero
    
    log_prob_sum = sum(math.log(1 / len(word)) for word in words)  # Approximate log probability
    perplexity = math.exp(-log_prob_sum / N)
    return perplexity

def benchmark_model(model, prompt):
    global inference_times, bleu_scores, perplexity_scores
    print(f"Starting benchmark for model: {model}")
    
    # Start system monitoring
    stop_event.clear()
    monitor_thread = threading.Thread(target=track_system_usage, args=(model,))
    monitor_thread.start()

    # Run Ollama model
    start_time = time.time()
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times[model] = inference_time
    
    # Stop monitoring
    stop_event.set()
    monitor_thread.join()

    generated_response = response.get("message", {}).get("content", "No response")
    bleu_score = sentence_bleu([REFERENCE_RESPONSE], generated_response.split())
    perplexity_score = calculate_perplexity(generated_response)
    
    bleu_scores[model] = bleu_score
    perplexity_scores[model] = perplexity_score
    
    print(f"Inference Time for {model}: {inference_time:.2f} seconds | BLEU Score: {bleu_score:.2f} | Perplexity: {perplexity_score:.2f}")
    return generated_response

def save_benchmark_plot():
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Benchmarking LLMs on {cpu_name} | RAM: {ram_total} GB", fontsize=12, fontweight="bold")

    # Plot CPU Usage
    for model in MODELS:
        plt.plot(timestamps[:len(cpu_usage[model])], cpu_usage[model], label=f"{model} - CPU Usage (%)")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.legend()

    # Add GPU usage note
    gpu_note = "Note: GPU not used in this benchmark"
    plt.figtext(0.5, 0.01, gpu_note, wrap=True, horizontalalignment='center', fontsize=10, color='gray')

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.abspath(f"{timestamp}_benchmark_results.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved benchmark results at: {save_path}")


# Main Execution
if __name__ == "__main__":
    test_prompt = "Explain artificial intelligence in simple terms."
    results = {}
    
    for model in MODELS:
        results[model] = benchmark_model(model, test_prompt)
    
    # Save the results plot
    save_benchmark_plot()
    
    # Save raw benchmark data
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({"models": MODELS, "inference_times": inference_times, "bleu_scores": bleu_scores, "perplexity_scores": perplexity_scores}, f, indent=4)
    print(f"📄 Saved raw benchmark data in {output_file}")
