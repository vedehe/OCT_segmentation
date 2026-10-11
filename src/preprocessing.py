import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion


image_path = (
    "/Users/vaideheedongre/OCT_Segmentation/"
    "data/raw/OCTA_3mm/116.bmp"
)


def process_oct(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape

    # 1. Mild anisotropic denoising
    processed = anisotropic_diffusion(
        image.astype(np.float32),
        niter=10,
        kappa=20,
        gamma=0.10,
        option=1
    )
    processed = np.clip(processed, 0, 255).astype(np.uint8)

   
    # Detect the upper retinal boundary
    smooth = cv2.GaussianBlur(image, (5, 5), 0)

    gradient = np.abs(
        np.diff(smooth.astype(np.float32), axis=0)
    )

    # Smooth gradient responses horizontally and vertically
    gradient = cv2.GaussianBlur(gradient, (5, 5), 0)

    top_start = int(0.30 * height)
    top_end = int(0.50 * height)

    upper = np.zeros(width, dtype=np.int32)

    # First column: choose the strongest transition
    scores = gradient[top_start:top_end, 0]
    upper[0] = top_start + np.argmax(scores)

    # Subsequent columns: favour strong transitions near
    # the previous column's detected boundary.
    max_jump = 3

    for x in range(1, width):
        previous_y = upper[x - 1]

        start = max(top_start, previous_y - max_jump)
        end = min(top_end, previous_y + max_jump + 1)

        scores = gradient[start:end, x]

        if len(scores) > 0:
            upper[x] = start + np.argmax(scores)
        else:
            upper[x] = previous_y

    # Smooth the final boundary
    upper = cv2.GaussianBlur(
        upper.astype(np.float32).reshape(1, -1),
        (31, 1),
        0
    ).ravel().astype(np.int32)


    # 3. Keep everything below the upper boundary.
    # Only pixels ABOVE the boundary become black.
    mask = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        mask[upper[x]:, x] = 255

    masked = cv2.bitwise_and(
        processed, processed, mask=mask
    )

    # 4. Display boundary and result
    boundary_view = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for x in range(width):
        y = upper[x]
        boundary_view[max(0, y-1):min(height, y+2), x] = [
            255, 0, 0
        ]

        # Save the processed image
    output_dir = (
        "/Users/vaideheedongre/OCT_Segmentation/"
        "data/processed/images"
    )

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)

    success = cv2.imwrite(output_path, masked)

    if not success:
        raise IOError(f"Could not save image to: {output_path}")

    print(f"Processed image saved to: {output_path}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original OCT")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(boundary_view)
    plt.title("Upper Boundary (Red)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(masked, cmap="gray", vmin=0, vmax=255)
    plt.title("Above Boundary Black")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


process_oct(image_path)
