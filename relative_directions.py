import datadocket as dd
import numpy as np
import pickle

# params
VARIATIONS = ["question", "synonym_change", "antonym_change", "paraphrase_change", "letter_change"]
SHAPES = ["o", "s", "d", "v", "p"]

def relative_directions(
    average_embeddings_file: str,
    pca_model_file: str,
    variations: list) -> bool:
    """
    """
    
    # load average embeddings
    average_embeddings = dd.load.Json(average_embeddings_file)

    # load PCA model
    with open(pca_model_file, "rb") as f:
        pca = pickle.load(f)

    def calculate_relative_angle_degrees(question_vec, variation_vec):
        """
        Calculate the angle of variation_vec relative to question_vec as the reference (0,0)
        Returns angle in degrees (0 to 360)
        """
        # Convert to numpy arrays
        question_vec = np.array(question_vec)
        variation_vec = np.array(variation_vec)
        
        # Calculate the difference vector (variation - question)
        diff_vec = variation_vec - question_vec
        
        # Calculate angle with x-axis (using first component as x-axis reference)
        # For high-dimensional vectors, we'll use the first two principal components
        # or project onto a 2D plane for angle calculation
        
        # Method 1: Use first two dimensions for angle calculation
        if len(diff_vec) >= 2:
            x_component = diff_vec[0]
            y_component = diff_vec[1]
        else:
            # Fallback: use first dimension and magnitude
            x_component = diff_vec[0]
            y_component = np.linalg.norm(diff_vec[1:]) if len(diff_vec) > 1 else 0
        
        # Calculate angle in degrees
        angle_rad = np.arctan2(y_component, x_component)
        angle_deg = np.degrees(angle_rad)
        
        # Convert to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg

    # start directions list
    directions = []

    # iterate over average embeddings
    for average_data in average_embeddings:
        question_directions = {}
        question_directions["id"] = average_data["id"]
        
        # Get the question vector as reference
        question_vec = np.array(average_data["question"])

        for variation in variations:
            if variation == "question":
                # Question is the reference point, so angle is 0
                question_directions[variation] = 0.0
            else:
                # Calculate angle relative to the question vector
                variation_vec = np.array(average_data[variation])
                angle_degrees = calculate_relative_angle_degrees(question_vec, variation_vec)
                question_directions[variation] = angle_degrees

        directions.append(question_directions)

    # save directions
    dd.save.Json("data/directions.json", directions, mode="w")

    # get average directions for each variation type
    average_directions = {}
    for variation in variations:
        list_directions = []
        for direction in directions:
            list_directions.append(direction[variation])
        
        # Calculate circular statistics for angles
        # Convert to radians for circular mean calculation
        angles_rad = np.radians(list_directions)
        
        # Calculate circular mean
        cos_mean = np.mean(np.cos(angles_rad))
        sin_mean = np.mean(np.sin(angles_rad))
        circular_mean_rad = np.arctan2(sin_mean, cos_mean)
        circular_mean_deg = np.degrees(circular_mean_rad)
        if circular_mean_deg < 0:
            circular_mean_deg += 360
        
        # Calculate circular standard deviation
        R = np.sqrt(cos_mean**2 + sin_mean**2)  # Resultant length
        circular_std_deg = np.degrees(np.sqrt(-2 * np.log(R)))
        
        average_directions[variation] = {
            "average": circular_mean_deg,
            "standard_deviation": circular_std_deg,
            "median": np.median(list_directions),
            "min": np.min(list_directions),
            "max": np.max(list_directions)
        }

    # save average directions
    dd.save.Json("data/relative_directions_stats.json", average_directions, mode="w")

    return True