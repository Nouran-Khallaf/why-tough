import numpy as np

# Define classes in the order of their encoded values
classes = [
    "Explanation",          # Encoded as 0
    "Grammatical Adjustments",  # Encoded as 1
    "Modulation",           # Encoded as 2
    "Omission",             # Encoded as 3
    "Substitution",         # Encoded as 4
    "Syntactic Changes",    # Encoded as 5
    "Transposition"         # Encoded as 6
]

# Save the class labels as a .npy file
np.save("label_classes.npy", np.array(classes))
print("label_classes.npy has been created successfully!")
