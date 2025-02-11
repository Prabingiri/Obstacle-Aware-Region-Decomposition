# Synthetic Data for Obstacle-Aware Partitioning

Welcome! This directory contains scripts and data for generating synthetic obstacle datasets. The goal is to provide a flexible environment where researchers can experiment with different obstacle configurations for obstacle-aware partitioning or related spatial analysis tasks.

---

## dataset overview


1. **iowa/**  
   Contains real-world GeoJSON files outlining Iowa’s boundaries and relevant FAA features.  

2. **synthetic_data/**  
   - **Non_Uniform_Synthetic_data_generator_clustering.py**  
     Creates synthetic obstacles under a non-uniform (cluster-based) distribution.  
   - **synthetic_data_generation_on_three_conditions.py**  
     Generates synthetic obstacles under various conditions (e.g., uniform, random) for comparison.  
   - **view_obstacles.py**  
     Loads the generated `.json` files and creates `.png` visualizations showing the obstacle layout.

3. **synthetic_data_generated/**  
   The JSON files and optional images (`.png`) produced by the generator scripts.  

4. **README.md**  
   This file, describing the directory’s purpose, usage, and structure.

5. **requirements.txt**  
   Lists the Python libraries needed to run these scripts.

---

## Installation & Dependencies

1. **Create and Activate a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate     # On macOS/Linux
   # or
   .\venv\Scripts\activate      # On Windows

2. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    
    
# How to Generate Synthetic Data

1. **Synthetic_data_generation_on_three_conditions.py**

    ****Location: dataset/synthetic_data/synthetic_data_generation_on_three_conditions.py****

    **Purpose:**
     Creates synthetic obstacle distributions under one or more preset “conditions” (uniform distribution, random distribution, etc.).

     **Usage:
Adjust parameters like region_size, obstacle_percentage, etc. at the bottom of the file.**
    ```bash
    cd dataset/synthetic_data
    python synthetic_data_generation_on_three_conditions.py

2. **Non_Uniform_Synthetic_data_generator_clustering.py**

   ****Location: dataset/synthetic_data/Non_Uniform_Synthetic_data_generator_clustering.py****
     **Purpose:**
     Generates synthetic obstacles in a rectangular region, potentially clustering them around specified centers.

     **Usage:**
Open the file and adjust parameters (e.g., region_size, obstacle_percentage, clusters) as needed.

   **Run:** 
   ```bash
    cd dataset/synthetic_data
    python Non_Uniform_Synthetic_data_generator_clustering.py
   
In both cases, Look in synthetic_data_generated/ (or your specified folder) for new .json files.

3. **Visualizing the Results**

   ****Location: dataset/synthetic_data/view_obstacles.py****
 
    **Usage:** Adjust the path in the script to point to your generated .json file

   **Run:** 
   ```bash
    cd dataset/synthetic_data
    python view_obstacles.py
   
A new .png file with the same base name as your JSON will be saved in the same directory, providing a visual layout of the obstacles.

5. **Tips & Customization**

**Clustering Logic:**
If you want more clusters or different distributions, modify or add to the clusters list in Non_Uniform_Synthetic_data_generator_clustering.py.
I would use clustering logic based on requirement.

**Geometry & Validations:**
Shapely ensures obstacles remain valid polygons. If you notice the generator stops placing obstacles early, it might be due to overlap or connectivity constraints.

**Connectivity:**
Some scripts enforce that removing obstacle areas must leave the region as a single connected polygon. If needed, you can relax or remove this rule in the code.


Note: To insert required libraries:
```bash
pip install -r requirements.txt