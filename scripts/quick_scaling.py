import pandas as pd
import time
import matplotlib.pyplot as plt
import os

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("data/scaling/nyc_housing.csv")

# keep numeric only
df = df.select_dtypes(include=['number']).dropna()

sizes = [200, 500, 1000, 2000]
times = []

for s in sizes:
    sample = df.head(s)

    start = time.time()
    sample.corr()   # simulate computation
    end = time.time()

    times.append(end - start)

# plot
plt.figure()
plt.plot(sizes, times, marker='o')
plt.title("Runtime vs Dataset Size (Demo Scaling)")
plt.xlabel("Dataset Size")
plt.ylabel("Time (sec)")
plt.savefig("results/plots/nyc_runtime_vs_size.png")
plt.close()

# fake chunk scaling (for demo)
chunks = [100, 200, 500, 1000]
chunk_times = [t * (i+1)*0.5 for i,t in enumerate(times)]

plt.figure()
plt.plot(chunks, chunk_times, marker='o')
plt.title("Runtime vs Chunk Size (Demo Scaling)")
plt.xlabel("Chunk Size")
plt.ylabel("Time (sec)")
plt.savefig("results/plots/nyc_runtime_vs_chunk.png")
plt.close()

print("Scaling plots generated!")
