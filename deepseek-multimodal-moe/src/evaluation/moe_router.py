# Redefine MoERouterMonitor and Plotting Function
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class MoERouterMonitor:
    def __init__(self):
        self.step = 0
        self.entropy_history = []
        self.capacity_counts = None
        self.num_experts = None
        self._handles = []

    @staticmethod
    def _entropy(probs, dim=-1, eps=1e-9):
        p = probs.clamp_min(eps)
        return -(p * p.log()).sum(dim=dim)

    def _hook(self, name):
        def fn(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x is None or not torch.is_tensor(x) or x.dim() < 2:
                return

            router_logits = x.float() # [*, 64]

            if self.num_experts is None:
                self.num_experts = router_logits.shape[-1] # This will now be 64
                self.capacity_counts = torch.zeros(self.num_experts, dtype=torch.long, device="cpu")

            probs = torch.softmax(router_logits, dim=-1)
            H = self._entropy(probs, dim=-1).mean().item()
            self.entropy_history.append((int(self.step), float(H)))

            top_k_indices = torch.topk(router_logits, k=2, dim=-1).indices.flatten()
            counts = torch.bincount(top_k_indices.to("cpu"), minlength=self.num_experts)
            self.capacity_counts[:len(counts)] += counts

        return fn

    def attach(self, model):
        # (The PEFT model adds 'base_model.model.' to the start)
        self.target_names = [
            f"base_model.model.model.layers.{i}.mlp.gate" for i in range(1, 28)
        ]

        print("--- Attaching MoE Monitor (THE REAL FIX) ---")
        print(f"Attempting to hook {len(self.target_names)} exact router names...")
        count = 0

        # We create a dictionary of all modules for fast lookup
        module_lookup = dict(model.named_modules())

        for name in self.target_names:
            if name in module_lookup:
                try:
                    module = module_lookup[name]
                    self._handles.append(module.register_forward_hook(self._hook(name)))
                    count += 1
                except Exception as e:
                    print(f"  [FAILED] to attach to {name}: {e}")
            else:
                print(f"  [NOT FOUND] Could not find module: {name}")

        print(f"\n--- Attached to {count} router module(s) ---")
        if count != 27:
            print("\nCRITICAL WARNING: Did not hook all routers. The names might be slightly different.")
            print("Please check the 'NOT FOUND' messages above.")
        else:
            print("Successfully hooked all required MoE routers.")

    def detach(self):
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles = []
        print("--- MoE Monitor detached ---")

    def tick(self):
        self.step += 1

def save_router_plots_and_stats(out_dir, router_monitor):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if router_monitor.entropy_history:
        xs = [s for s,_ in router_monitor.entropy_history]
        ys = [h for _,h in router_monitor.entropy_history]
        plt.figure(); plt.plot(xs, ys)
        plt.title("Routing Entropy (mean per token)"); plt.xlabel("Step"); plt.ylabel("Entropy")
        plt.tight_layout(); plt.savefig(out_dir/"routing_entropy.png"); plt.close()

    if router_monitor.capacity_counts is not None and router_monitor.num_experts:
        x = np.arange(router_monitor.num_experts)
        y = router_monitor.capacity_counts.numpy()
        total_calls = y.sum()
        y_percent = (y / total_calls) * 100 if total_calls > 0 else y

        plt.figure(figsize=(max(10, router_monitor.num_experts // 2), 6)) # Made wider for 64 experts
        plt.bar(x, y_percent)
        plt.title(f"Expert Capacity Utilization (Top-2, based on {total_calls} calls)")
        plt.xlabel("Expert ID"); plt.ylabel("Utilization (%)")
        plt.xticks(x, rotation=90, fontsize=8) # Rotate labels for 64 experts
        plt.tight_layout(); plt.savefig(out_dir/"expert_capacity.png"); plt.close()

    stats = {
        "entropy_history": router_monitor.entropy_history,
        "num_experts": router_monitor.num_experts,
        "capacity_counts": router_monitor.capacity_counts.tolist() if router_monitor.capacity_counts is not None else None
    }
    with open(out_dir/"router_stats.json","w") as f:
        json.dump(stats, f, indent=2)

    print(f" Routing plots and stats saved to: {out_dir}")

print("MoERouterMonitor class done.")

# Run Evaluation with MoE Monitor Attached

# 1. Instantiate the (new, real) monitor
router_monitor = MoERouterMonitor()

# 2. Attach it to our existing evaluation model
try:
    router_monitor.attach(eval_model)
except NameError:
    print("ERROR: 'eval_model' is not in memory.")

# 3. We use the DataLoader with batch_size=1
try:
    eval_dataloader
except NameError:
    print("Re-creating eval_dataloader...")
    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=1,
        collate_fn=data_collator
    )

# 4. Run the manual loop again (this populates the monitor)
eval_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Running Evaluation to Capture MoE Metrics ---")

with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc="Monitoring MoE Routing"):
        router_monitor.tick()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = eval_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# 5. Detach the monitor (cleanup)
router_monitor.detach()

print("\n--- MoE Data Capture Complete ---")
print(f"Monitor captured {len(router_monitor.entropy_history)} steps.")
print(f"Total expert calls recorded: {router_monitor.capacity_counts.sum().item() if router_monitor.capacity_counts is not None else 'N/A'}")

Generate MoE Metrics and Plots
from IPython.display import Image, display

# 1. Define the output directory
moe_output_dir = "./content/moe-metrics-real"   #adjust the path accoridingly

# 2. Call the plotting function
print(f"--- Saving MoE Metrics to {moe_output_dir} ---")
try:
    save_router_plots_and_stats(moe_output_dir, router_monitor)

    print("\n--- Summary of Captured Metrics (THE REAL ONE) ---")
    print(f"Total Experts Detected: {router_monitor.num_experts}") 
    if router_monitor.entropy_history:
        ys = [h for _,h in router_monitor.entropy_history]
        mean_entropy = np.mean(ys)
        print(f"Mean Routing Entropy: {mean_entropy:.4f}")

except Exception as e:
    print(f"\nAn error occurred during plotting: {e}")

# 3. Display the new, correct plots
entropy_img = Path(moe_output_dir) / "routing_entropy.png"
capacity_img = Path(moe_output_dir) / "expert_capacity.png"

if entropy_img.exists():
    print("\n--- Routing Entropy Plot (THE REAL ONE) ---")
    display(Image(filename=str(entropy_img)))
else:
    print("\nCould not find routing_entropy.png")

if capacity_img.exists():
    print("\n--- Expert Capacity Utilization Plot ---")
    display(Image(filename=str(capacity_img)))
else:
    print("\nCould not find expert_capacity.png")
