import os
import urllib.request
from tqdm import tqdm

# SAM model download configuration
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
WEIGHTS_DIR = "weights"
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_sam_weights():
    """Download SAM ViT-H checkpoint if not already present."""
    
    # Create weights directory if it doesn't exist
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(SAM_CHECKPOINT):
        print(f"✓ SAM checkpoint already exists at: {SAM_CHECKPOINT}")
        file_size = os.path.getsize(SAM_CHECKPOINT) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
        return
    
    print(f"Downloading SAM ViT-H checkpoint...")
    print(f"Source: {SAM_URL}")
    print(f"Destination: {SAM_CHECKPOINT}")
    print(f"This may take several minutes (~2.5 GB)...\n")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='SAM ViT-H') as t:
            urllib.request.urlretrieve(
                SAM_URL,
                SAM_CHECKPOINT,
                reporthook=t.update_to
            )
        
        print(f"\n✓ Download complete!")
        file_size = os.path.getsize(SAM_CHECKPOINT) / (1024 * 1024)  # MB
        print(f"  Saved to: {SAM_CHECKPOINT}")
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        # Clean up partial download
        if os.path.exists(SAM_CHECKPOINT):
            os.remove(SAM_CHECKPOINT)
        raise

if __name__ == "__main__":
    print("=" * 60)
    print("SAM (Segment Anything) Model Downloader")
    print("=" * 60)
    print()
    
    download_sam_weights()
    
    print()
    print("=" * 60)
    print("Setup complete! You can now run: python __main__.py")
    print("=" * 60)
