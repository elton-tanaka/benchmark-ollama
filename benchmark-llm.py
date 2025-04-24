import ollama
import psutil
import time
import threading
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import math

# Setup BLEU
nltk.download('punkt')
nltk.download('punkt_tab')
smooth_fn = SmoothingFunction().method4

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_available = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
except Exception as e:
    print(f"⚠️ Failed to initialize GPU monitoring: {e}")
    gpu_available = False
    gpu_name = None

# Define models to benchmark
MODELS = [
#    "qwen2.5:14b",  # Example of a standard model
    "deepseek-r1:14b",  # Example of a compressed model
#    "deepseek-r1:7b",
#    "qwen2.5:7b"
]

# Get system hardware info
cpu_name = f"{psutil.cpu_freq().max:.0f} MHz" if psutil.cpu_freq() else "Unknown CPU"
ram_total = round(psutil.virtual_memory().total / (1024**3), 2)
cpu_cores = psutil.cpu_count(logical=False)
cpu_threads = psutil.cpu_count(logical=True)

# Initialize storage
timestamps = []
cpu_usage = {model: [] for model in MODELS}
ram_usage = {model: [] for model in MODELS}
gpu_usage = {model: [] for model in MODELS} if gpu_available else {}
inference_times = {}
bleu_scores = {}
perplexity_scores = {}
peak_cpu_usage = {}
token_counts = {}
stop_event = threading.Event()

# Reference response
REFERENCE_SET = []
REFERENCE_SET.append("In recent years, the adoption of AI model‑compression techniques has made it possible to run advanced neural networks on edge devices. By combining parameter pruning, weight quantization, and knowledge distillation, researchers have cut memory usage by as much as 80 % with no significant loss in accuracy. This progress paves the way for computer‑vision and natural‑language‑processing applications on smartphones, drones, and industrial sensors, removing the need for a constant cloud connection.")

# Text cleaner
def clean_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return word_tokenize(text)

# Approximate perplexity
def calculate_perplexity(text):
    tokens = clean_and_tokenize(text)
    N = len(tokens)
    if N == 0:
        return float('inf')
    log_prob_sum = sum(math.log(1 / len(token)) for token in tokens)
    return math.exp(-log_prob_sum / N)

# Track system usage
def track_system_usage(model):
    while not stop_event.is_set():
        timestamps.append(time.time())
        usage = psutil.cpu_percent(interval=0.5)
        cpu_usage[model].append(usage)
        ram_usage[model].append(psutil.virtual_memory().percent)
        if gpu_available:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            gpu_usage[model].append(util.gpu)

# Benchmark a single model
def benchmark_model(model, prompt):
    global inference_times, bleu_scores, perplexity_scores, peak_cpu_usage, token_counts

    print(f"🔍 Benchmarking model: {model}")
    stop_event.clear()
    monitor_thread = threading.Thread(target=track_system_usage, args=(model,))
    monitor_thread.start()

    start_time = time.time()
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    inference_time = end_time - start_time
    inference_times[model] = inference_time
    peak_cpu_usage[model] = max(cpu_usage[model]) if cpu_usage[model] else 0

    generated_response = response.get("message", {}).get("content", "No response")
    token_count = len(generated_response.split())
    token_counts[model] = token_count

    # Clean and compute BLEU
    all_references = []   # [[ref_tokens1, ref_tokens2, ...], ...]
    all_hypotheses = []   # [hyp_tokens1, hyp_tokens2, ...]

    for ref_text in REFERENCE_SET:
        ref_tokens = clean_and_tokenize(ref_text)
        hyp_tokens = clean_and_tokenize(generated_response)

        all_references.append([ref_tokens])          # wrap each ref in its own list
        all_hypotheses.append(hyp_tokens)

    bleu = corpus_bleu(
        all_references,
        all_hypotheses,
        smoothing_function=smooth_fn
    )
    bleu_scores[model] = bleu

    # Compute Perplexity
    perplexity = calculate_perplexity(generated_response)
    perplexity_scores[model] = perplexity

    print(f"✅ Inference Time: {inference_time:.2f}s | BLEU: {bleu:.2f} | Perplexity: {perplexity:.2f} | Tokens: {token_count} | Peak CPU: {peak_cpu_usage[model]:.2f}%")
    return generated_response

# Save performance plot
def save_benchmark_plot(timestamp):
    total_plots = 3 if gpu_available else 2
    plt.figure(figsize=(10, 4 * total_plots))

    title = f"Benchmarking LLMs on {cpu_name} | RAM: {ram_total} GB"
    if gpu_available:
        title += f" | GPU: {gpu_name}"
    plt.suptitle(title, fontsize=12, fontweight="bold")

    # CPU Plot
    ax1 = plt.subplot(total_plots, 1, 1)
    for model in MODELS:
        ax1.plot(timestamps[:len(cpu_usage[model])], cpu_usage[model], label=f"{model} - CPU")
    ax1.set_ylabel("CPU Usage (%)")
    ax1.legend()

    # RAM Plot
    ax2 = plt.subplot(total_plots, 1, 2)
    for model in MODELS:
        ax2.plot(timestamps[:len(ram_usage[model])], ram_usage[model], label=f"{model} - RAM")
    ax2.set_ylabel("RAM Usage (%)")
    ax2.legend()

    # GPU Plot
    if gpu_available:
        ax3 = plt.subplot(total_plots, 1, 3)
        for model in MODELS:
            ax3.plot(timestamps[:len(gpu_usage[model])], gpu_usage[model], label=f"{model} - GPU")
        ax3.set_ylabel("GPU Usage (%)")
        ax3.legend()
    else:
        plt.figtext(0.5, 0.01, "Note: GPU not used in this benchmark", ha='center', fontsize=10, color='gray')

    save_path = os.path.abspath(f"benchmark_results_{timestamp}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved plot at: {save_path}")

# Save separated comparison charts in a single figure
def save_comparison_chart(timestamp):
    labels = MODELS
    x = range(len(labels))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.bar(labels, [inference_times[m] for m in labels], color='steelblue')
    ax1.set_title('Inference Time (s)')
    ax1.set_ylabel('Seconds')

    ax2.bar(labels, [bleu_scores[m] for m in labels], color='orange')
    ax2.set_title('BLEU Score')
    ax2.set_ylabel('Score')

    ax3.bar(labels, [perplexity_scores[m] for m in labels], color='green')
    ax3.set_title('Perplexity')
    ax3.set_ylabel('Score')

    fig.suptitle("Model Benchmark Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.abspath(f"comparison_summary_{timestamp}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved comparison chart at: {save_path}")

# Main execution
if __name__ == "__main__":
    prompt = "Task: Translate the following Brazilian‑Portuguese passage into natural, idiomatic English. Text (PT‑BR): Nos últimos anos, a adoção de técnicas de compressão de modelos de inteligência artificial tem permitido executar redes neurais avançadas em dispositivos de borda. Ao combinar poda de parâmetros, quantização de pesos e destilação de conhecimento, pesquisadores reduziram o consumo de memória em até 80 % sem perda significativa de precisão. Essa evolução abre caminho para aplicações de visão computacional e processamento de linguagem natural em smartphones, drones e sensores industriais, eliminando a dependência de conexões constantes com a nuvem."
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in MODELS:
        results[model] = benchmark_model(model, prompt)

    save_benchmark_plot(timestamp)
    save_comparison_chart(timestamp)

    # Save raw results
    data = {
        "models": MODELS,
        "inference_times": inference_times,
        "bleu_scores": bleu_scores,
        "perplexity_scores": perplexity_scores,
        "token_counts": token_counts,
        "peak_cpu_usage": peak_cpu_usage,
        "hardware": {
            "cpu": cpu_name,
            "cpu_cores": cpu_cores,
            "cpu_threads": cpu_threads,
            "ram_gb": ram_total,
            "gpu": gpu_name if gpu_available else "None"
        }
    }

    result_file = f"benchmark_results_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Benchmarking complete. Results saved to {result_file}")
