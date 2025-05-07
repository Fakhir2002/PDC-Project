//python3 split.py amazon400.txt 2

import sys

if len(sys.argv) != 3:
    print("Usage: python3 partition_txt.py amazon400.txt 2")
    sys.exit(1)

input_file = sys.argv[1]
num_parts = int(sys.argv[2])

output_files = [open(f"part{i}.txt", "w") for i in range(num_parts)]

# Choose strategy: "round_robin" or "range"
strategy = "round_robin"

# Optional: get max node ID to do range-based partitioning
max_node = 0
with open(input_file) as f:
    for line in f:
        u, v, w = map(int, line.strip().split())
        max_node = max(max_node, u)

cutoff = max_node // num_parts

# Partition edges
with open(input_file) as f:
    for line in f:
        u, v, w = map(int, line.strip().split())
        if strategy == "round_robin":
            part_id = u % num_parts
        elif strategy == "range":
            part_id = u // (cutoff + 1)
        else:
            raise ValueError("Unknown strategy.")
        output_files[part_id].write(f"{u} {v} {w}\n")

for f in output_files:
    f.close()

print("✅ Partitioning complete: part0.txt, part1.txt created.")
