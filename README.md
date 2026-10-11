# OCT Retinal Layer Segmentation

U-Net based retinal layer segmentation from OCT B-scans.

## Project Structure

- `data/raw` - original raw datasets
- `data/images` - OCT B-scan images
- `data/masks` - manually labeled segmentation masks
- `src/model.py` - U-Net architecture
- `src/dataset.py` - dataset loading
- `src/train.py` - training
- `src/predict.py` - inference
- `src/losses.py` - loss functions
- `src/metrics.py` - evaluation metrics
- `outputs/` - model outputs and checkpoints

## Training

```bash
python src/train.py