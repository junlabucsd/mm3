## Focus tracking workflow

### 1. mm3_Foci.py
Finds foci using a Laplacian convolution. See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html for details.
Foci are linked to the cell objects in which they appear.
Input: .pkl file of cell objects from mm3_Track-Standard.py
Output: .pkl file with a dictionary of cell objects, with x, y and time positions of foci detections stored as attributes of the corresponding cell objects.

### 2. mm3_TrackFoci.py
Constructs a tree of replication cycles from individual focus detections.
Output: .pkl file with a dictionary of replication trace objects, with attributes initiation time, termination time, etc.

### 3. mm3_CurateFocusTracks.py
Allows manual curation of replication traces.

### 4. mm3_OutputCC.py
Calculation of cell cycle parameters.
