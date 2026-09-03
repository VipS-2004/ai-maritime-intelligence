import numpy as np
import cv2
import matplotlib.pyplot as plt



def generate_heatmap(results, orig_img):
    """Overlay a Gaussian density heatmap on the original image."""

    h, w, _ = orig_img.shape

    heatmap = np.zeros((h, w), dtype=np.float32)

    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.circle(heatmap, (cx, cy), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        orig_img,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    

    return overlay


def zone_based_analysis(results, orig_img, grid_size=4):
    """
    Divide image into a grid and count ships per zone.

    Highlights zones by density and identifies hotspots.

    Returns:
        zone_counts array
        hotspot coordinates
    """

    h, w, _ = orig_img.shape

    zone_h = h // grid_size
    zone_w = w // grid_size

    zone_counts = np.zeros(
        (grid_size, grid_size),
        dtype=int
    )

    # Assign each ship to a grid zone
    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = map(
            int,
            box.cpu().numpy()
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        grid_x = min(
            cx // zone_w,
            grid_size - 1
        )

        grid_y = min(
            cy // zone_h,
            grid_size - 1
        )

        zone_counts[grid_y][grid_x] += 1

    # Draw zone overlay
    overlay = orig_img.copy()

    for i in range(grid_size):

        for j in range(grid_size):

            x1 = j * zone_w
            y1 = i * zone_h

            x2 = x1 + zone_w
            y2 = y1 + zone_h

            count = zone_counts[i][j]

            # Green → Yellow → Red
            if count == 0:
                color = (0, 255, 0)

            elif count < 3:
                color = (0, 255, 255)

            else:
                color = (0, 0, 255)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                overlay,
                str(count),
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

   

    # Find hotspot zones
    max_count = np.max(zone_counts)

    hotspots = (
        np.argwhere(zone_counts == max_count)
        if max_count > 0
        else []
    )

    return zone_counts, hotspots, overlay  


def show_detection_results(results):
    """
    Display YOLOv8 detection results.
    """

    img = results[0].plot()

    plt.figure(figsize=(10, 7))

    plt.imshow(
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")

    plt.title("YOLOv8 Ship Detection")

    plt.tight_layout()

    plt.show()

    return img
