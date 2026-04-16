from bing_image_downloader import downloader
import os

# Your categories
categories = [
    "avocado inside ripe",
    "avocado outside ripe",
    "avocado inside unripe",
    "avocado outside unripe",
    "avocado green",
    "avocado mushy",
    "avocado spoiled"
]

print("Starting the Machine Learning Data Scraper...")

for category in categories:
    # We still use your clever trick to name the folder!
    folder_name = category.split()[0]  
    
    print(f"\n--- Downloading images for: {category} ---")
    
    # This single line replaces your entire download function
    downloader.download(
        category, 
        limit=20,                    # Number of images to grab
        output_dir=f'data/{folder_name}', # Where to save them
        adult_filter_off=False,       # Keep it safe
        force_replace=False,          # Don't redownload if we already have them
        timeout=60                    # Give up if a site is too slow
    )

print("\nDataset collection complete!")