
import torch
import torch.nn as nn
import torch.optim as optim

from data.data_loader import get_data_loaders
from models.mnist import mnist
from utils.training import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders()

# Model
model = mnist().to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")




from utils.metrics import calculate_sparsity, quantization_stats

def detailed_sparsity_report(model):
    sparsity = calculate_sparsity(model)
    print(f"Sparsity: {sparsity:.2f}%")

print("Baseline Accuracy:", evaluate(model, test_loader, device))


# Apply Pruning
PRUNE_RATIO = 0.5
model.prune(PRUNE_RATIO)
sparsity = calculate_sparsity(model)
print(f"Model Sparsity: {sparsity:.2f}%")




# Fine Tuning after Pruning
for epoch in range(3):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)

    print(f"[Finetune] Epoch {epoch+1}, Loss: {loss:.4f}, Acc: {acc:.2f}%")

print("Before Quantization Accuracy:", evaluate(model, test_loader, device))


# APPLY QUANTIZATION
NUM_BITS = 8

model.quantize(NUM_BITS)

print("After Quantization Accuracy:", evaluate(model, test_loader, device))




from utils.logger import print_header
from utils.huffman_utils import apply_huffman_to_npz
from utils.metrics import calculate_sparsity, detailed_sparsity, quantization_stats, compute_storage

# ===============================
print_header("Deep Compression Pipeline v1.0")

print("[*] Computation Device:", device)

# ===============================
print("\n[*] Loading MNIST Dataset...")
print("[*] Initializing Baseline MNIST_MLP Model...")

# ===============================
print("\n[*] Starting Baseline Training (5 Epochs)...")


print_header("PHASE 2: PRUNING")

print("\n--- BEFORE PRUNING ---")
detailed_sparsity(model)   # don't assign here

print(f"\n[*] Applying Magnitude Pruning...")
model.prune(PRUNE_RATIO)

print("\n--- AFTER PRUNING ---")
sparsity = detailed_sparsity(model)

print(f"\n[*] Global Sparsity achieved: {sparsity:.2f}%")


print_header("PHASE 3: QUANTIZATION")

print("\n[*] Before Quantization:")
quantization_stats(model)

print("\n[*] Applying 8-bit Quantization (K-Means Weight Sharing)...")
print("    Targeting 256 unique weight values per layer.")

model.quantize(8)

print("\n[*] After Quantization:")
quantization_stats(model)



print("\n[*] --- STORAGE SAVINGS ---")

baseline_size = compute_storage(model, 32)
compressed_size = compute_storage(model, 8)

print(f"   [-] Baseline Storage Size: {baseline_size:.4f} MB (32-bit)")
print(f"   [-] Compressed Storage Size: {compressed_size:.4f} MB (8-bit)")
print(f"   [-] Compression Ratio: {baseline_size/compressed_size:.1f}x smaller!")


# -------------------------------
# QUANTIZATION DONE
# -------------------------------

print("\n[*] Evaluating Final Compressed Model...")
acc = evaluate(model, test_loader, device)
print(f"==> Test Accuracy: {acc:.2f}%")

from utils.serialization import save_compressed_model
import os

os.makedirs("compressed_models", exist_ok=True)

save_compressed_model(model, "compressed_models/model.npz")

print("\n[*] Model saved for compression stage")



print_header("PHASE 4: HUFFMAN ENCODING")
from utils.huffman_utils import apply_huffman_to_npz

ratio, huff_size = apply_huffman_to_npz("compressed_models/model.npz")

print(f"\n   [-] Huffman Encoded Size: {huff_size:.4f} MB")
print(f"\n   [!!!] FINAL COMPRESSION RATIO: {ratio:.1f}x smaller [!!!]")





# Here we Check compression effect
from utils.metrics import count_unique_weights
unique_vals = count_unique_weights(model)
print("Unique weight values:", unique_vals)

for module in model.modules():
    print(type(module))




def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("Total Parameters:", count_params(model))


# import psutil
# import os
# process = psutil.Process(os.getpid())
# print("RAM (MB):", process.memory_info().rss / 1024**2)





