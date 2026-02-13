"""Retrieve eval artifacts from the latest ZenML pipeline run."""
from zenml.client import Client

c = Client()

# Rush hour plot
rush_art = c.get_artifact_version("rush_hour_plot_html")
print("rush_hour_plot_html:")
print(f"  Type: {rush_art.type}")
print(f"  URI: {rush_art.uri}")
html = rush_art.load()
print(f"  Size: {len(html)} chars")
with open("/tmp/rush_hour_plot.html", "w") as f:
    f.write(html)
print("  Saved to: /tmp/rush_hour_plot.html")

# Interpretation
interp_art = c.get_artifact_version("interpretation_html")
print("\ninterpretation_html:")
print(f"  Type: {interp_art.type}")
print(f"  URI: {interp_art.uri}")
content = interp_art.load()
print(f"  Size: {len(content)} chars")
with open("/tmp/interpretation.html", "w") as f:
    f.write(content)
print("  Saved to: /tmp/interpretation.html")

# Metrics
mae = c.get_artifact_version("test_mae").load()
smape = c.get_artifact_version("test_smape").load()
print(f"\ntest_mae: {mae}")
print(f"test_smape: {smape}")
