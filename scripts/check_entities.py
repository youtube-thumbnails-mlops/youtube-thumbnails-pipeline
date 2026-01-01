import wandb

# api.viewer is an object property in some versions, or a method in others?
# Let's use the safe attributes on api directly
print(f"👤 Current User: {api.viewer.username}")
print(f"🏢 Default Entity: {api.default_entity}")
print("👥 Teams/Entities available:")
# We can infer entities by checking projects?
pass
print("👥 Teams/Entities available:")

# There isn't a direct .teams() on viewer in all SDK versions, 
# but projects usually belong to an entity.
# We can try to list projects for the default entity.

try:
    teams = viewer.teams() # Some SDKs support this
    for t in teams:
        print(f"   - {t.name}")
except:
    print("   [Could not fetch teams directly]")

