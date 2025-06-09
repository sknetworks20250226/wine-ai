import kagglehub

# Download latest version
path = kagglehub.dataset_download("christopheiv/winemagdata130k")

print("Path to dataset files:", path)