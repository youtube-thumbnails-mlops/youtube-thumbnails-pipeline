import wandb

api = wandb.Api()
print("🔍 Listing All Projects for 'daniele5':")
projects = api.projects("daniele5")
for p in projects:
    print(f"   📂 Project: {p.name}")
