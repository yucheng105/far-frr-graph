'''
# Multi-Detector Double Thresholding Architecture
- This module is designed to compute all possible combinations of thresholds for a multi-detector system with double thresholding.
- all calculated results are stored in results
- each combination of detectors would have its own directory to store the results
    - eg. sbi->xception would have its own directory named sbi-xception/
- the results will be saved in CSV files according to the dataset used for testing

# CSVs
- The results will be saved in CSV files for each combination of detectors.
- Headers for the CSV files will include:
    - #detecor1_low, #detector1_high, #detector2_low, #detector2_high, ..., #detectorN_low, #detectorN_high, FAR, FRR

# FAR and FRR calculation
- FAR = 
    
'''