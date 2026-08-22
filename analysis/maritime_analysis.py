from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist

from configs.ship_classes import (
    MILITARY_CLASSES,
    CIVILIAN_CLASSES,
    ALL_SHIP_CLASSES
)

def get_ship_counts(results, model):
    """Count total ships and break down by class name."""
    
    boxes = results[0].boxes
    
    classes = boxes.cls.cpu().numpy()
    
    total = len(classes)
    
    class_counts = Counter(classes)

    named_counts = {
        model.names[int(k)]: v
        for k, v in class_counts.items()
    }

    return total, named_counts, classes


def congestion_level(count):
    """Classify traffic congestion based on total ship count."""
    
    if count < 5:
        return "Low"
    
    elif count < 15:
        return "Medium"
    
    else:
        return "High"


def classify_military_civilian(classes, model):
    """Split detected ships into military and civilian counts."""

    military = 0
    civilian = 0
    unknown = 0

    for cls in classes:

        name = model.names[int(cls)]

        if name in MILITARY_CLASSES:

            military += 1

        elif name in CIVILIAN_CLASSES:

            civilian += 1

        else:

            unknown += 1

    return military, civilian, unknown



def risk_level(military, total):
    """
    Assess risk based on military ship ratio.
    
        Low    → no military ships
        Medium → military ships < 30% of total
        High   → military ships >= 30% of total
    """

    if military == 0:
        return "Low"

    elif military < total * 0.3:
        return "Medium"

    else:
        return "High"
    


def check_unusual_clustering(results, threshold=50):
    """
    Detect if ships are suspiciously close together.
    Computes pairwise distances between ship centers.
    """

    centers = []

    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = box.cpu().numpy()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        centers.append([cx, cy])

    centers = np.array(centers)

    if len(centers) < 2:
        return False, "Not enough ships to check clustering."

    distances = pdist(centers)

    if np.min(distances) < threshold:

        return True, (
            f"Ships are closely clustered "
            f"(min distance: {np.min(distances):.1f}px)"
        )

    return False, "Normal distribution."    


def alert_system(military_count, total_ships, congestion):
    """
    Generate a system alert based on current maritime situation.
    """

    if military_count > 0 and congestion == "High":

        return (
            "CRITICAL ALERT: "
            "Military ships in high congestion zone!"
        )

    elif military_count > 0:

        return "WARNING: Military presence detected."

    elif congestion == "High":

        return "WARNING: Heavy maritime traffic."

    else:

        return "Normal maritime activity."


def density_label(total_ships):
    """
    Classify maritime traffic density.
    """

    if total_ships < 5:
        return "Sparse"

    elif total_ships < 15:
        return "Moderate"

    else:
        return "Dense"

def generate_analysis(results, model):
    """
    Generate a complete structured maritime analysis.
    """

    # 1. Ship counts
    total, class_breakdown, classes = get_ship_counts(results, model)

    # 2. Congestion
    congestion = congestion_level(total)

    # 3. Military / civilian classification
    military, civilian, unknown = classify_military_civilian(
        classes,
        model
    )

    # 4. Risk
    risk = risk_level(military, total)

    # 5. Clustering
    clustering_detected, clustering_message = check_unusual_clustering(
        results
    )

    # 6. Alert
    alert = alert_system(
        military,
        total,
        congestion
    )

    # 7. Density
    density = density_label(total)

    # Return everything in one structured object
    return {
        "total_ships": total,
        "class_breakdown": class_breakdown,

        "congestion": {
            "level": congestion,
            "density": density
        },

        "military_ships": military,
        "civilian_ships": civilian,
        "unknown_ships": unknown,

        "risk_level": risk,

        "clustering": {
            "detected": clustering_detected,
            "message": clustering_message
        },

        "alert": alert
    }

