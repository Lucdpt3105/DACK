import kagglehub

# Download latest version
path = kagglehub.dataset_download("maitam/vietnamese-traffic-signs")

print("Path to dataset files:", path)