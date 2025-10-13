import matplotlib.pyplot as plt

# Set global font family and size
plt.rcParams["font.family"] = "DejaVu Sans"  # change to "Arial", "Times New Roman", etc.
plt.rcParams["font.size"] = 16  # global font size

# Data
nodes = [1, 2, 4, 8, 16, 32]
ray_data = [3.405619686, 3.056751687, 20.48397248, 26.09462115, 33.42109212, 40.09736109]
ray_original = [9.219070767, 13.43074186, 17.95581079, 23.16180815, 23.3229512, 21.18512729]
ray_original_streaming = [9.495453187, 12.47886382, 20.91097013, 20.34612821, 24.29092928, 22.09533026]

# Plot with log scale for x-axis to give equal distance for powers of 2
plt.figure(figsize=(5, 4))
plt.plot(nodes, ray_data, marker='o', label="Radar")
plt.plot(nodes, ray_original, marker='s', label="Ray")
plt.plot(nodes, ray_original_streaming, marker='^', label="Ray Generator")

# Labels and legend
plt.xscale("log", base=2)
plt.xticks(nodes, nodes)
plt.yticks([10, 20, 30, 40, 50, 60])  # enforce y-ticks
plt.xlabel("Number of Nodes")
plt.ylabel("Throughput (GB/s)")
# plt.title("Throughput vs Number of Clusters (Log Scale X-axis)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig("scalability_plot.pdf")
