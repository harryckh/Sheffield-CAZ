# Assessing the Impact of Sheffield's Clean Air Zone on Air Quality and Traffic Volume

Notes
=======================

1. This repository contains the code and data used in the paper 
	"Assessing the Impact of Sheffield's Clean Air Zone on Air Quality and Traffic Volume".
2. This code is written in Python.



Usage
=======================

Step 1: Open the Complete_Analysis.ipynb
You will find five different sections in this notebook

1. AQ Data Preparation
2. AQ Changes
3. DiD Analysis
4. Traffic Data Preparation and Changes
5. Traffic Correlation

Step 2: Set the folder_path to specify the working directory.

Step 3: You might need to install some required packages through
	
	!pip install osmnx
	!pip install geopandas
	!pip install shapely

Step 3: (Optional) Run the data pre-processing steps on the raw data. Note that pre-processed data is also included.

Step 4: Perform the following analysis steps:

(a) Difference in differences on Air quality data

(b) Wilcoxon Signed-Rank Test on Traffic data

(c) Corrleation Test on Air quality and traffic data

Step 5. Collect the corresponding results, statistics and charts

