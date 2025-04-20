import matplotlib.pyplot as plt

SYSTEM_NAME = "Radar"

def plot_completion_time_with_staged(data_dict, output_dir, title="RAG Benchmark"):
    FONT_SIZE = 18

    # Staged batch baseline (1 GPU) in minutes
    # staged_batch_times = {
    #     "Encoding": 410.20 / 60,
    #     "Retrieval": 343.34 / 60,
    #     "Generation": 3051.68 / 60
    # }
    staged_batch_times = {
        "Encoding": 1093.43 / 60,
        "Retrieval": 740.68 / 60,
        "Generation": 7713.11 / 60
    }
    

    # Sort GPU bars
    sorted_items = sorted(data_dict.items(), key=lambda x: int(x[0].split()[0]))
    gpu_labels, gpu_times = zip(*sorted_items)

    # Prepare all labels and bar values
    labels = ["1 GPU\nStaged Batch"] + list(gpu_labels)
    bar_positions = list(range(len(labels)))

    # Stacked bar for staged_batch
    staged_colors = {
        "Encoding": "#b2df8a",
        "Retrieval": "#fdbf6f",
        "Generation": "#fb9a99"
    }
    staged_bottom = 0
    plt.figure(figsize=(10, 6))
    for stage, color in staged_colors.items():
        time = staged_batch_times[stage]
        bar = plt.bar(0, time, bottom=staged_bottom, color=color, label=stage)

        # Annotate inside the bar segment
        y_center = staged_bottom + time / 2
        plt.text(
            0, y_center, f"{time:.1f}", ha='center', va='center', fontsize=FONT_SIZE-1
        )
        staged_bottom += time

    # Bars for the original RAG pipeline (1-4 GPU)
    bars = plt.bar(bar_positions[1:], gpu_times, color="#A6CEE3", label=SYSTEM_NAME)
    

    # Vertical dashed line between staged_batch and normal pipeline
    plt.axvline(x=1.5, color="black", linestyle="--", linewidth=1)

    # Y-axis formatting
    plt.ylim(0, 170)
    plt.yticks(range(0, 161, 20), fontsize=FONT_SIZE)
    plt.xticks(bar_positions, labels, fontsize=FONT_SIZE)

    # Titles & labels
    # plt.title(title, fontsize=FONT_SIZE)
    # plt.xlabel("Configuration", fontsize=FONT_SIZE)
    plt.ylabel("Job Completion Time (minutes)", fontsize=FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels above bars
    for i, time in enumerate([sum(staged_batch_times.values())] + list(gpu_times)):
        plt.text(i, time + 0.3, f"{time:.1f}", ha='center', va='bottom', fontsize=FONT_SIZE)

    # Legend
    plt.legend(fontsize=FONT_SIZE - 2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/job_completion_time_v2.pdf")

# Example usage
# data = {
#     f"1 GPU\n{SYSTEM_NAME}": 49.5,
#     "2 GPU": 26.0,
#     "3 GPU": 18.4,
#     "4 GPU": 14.4
# }
data = {
    f"1 GPU\n{SYSTEM_NAME}": 7226.16 / 60,
    "2 GPU": 3835.33 / 60,
    "4 GPU": 2016.42 / 60,
    "8 GPU": 1119.92 / 60,
}
output_dir = "/m-coriander/coriander/yilegu/ray/ray/logs/finalized"
plot_completion_time_with_staged(data, output_dir=output_dir)
