# Zillow Data Organization Project

## Overview
This project collects and organizes various real estate data points from Zillow into a comprehensive CSV file named `Zillow_Data.csv`. It processes multiple data sources to create a unified view of Zillow housing data for metro areas, states, and the United States.

## Requirements
The script requires the following Python packages:
- pandas
- numpy
- requests
- beautifulsoup4
- selenium
- undetected-chromedriver
- selenium-stealth

You can install these packages using:
```
pip install -r requirements.txt
```

## Data Sources
The script processes multiple CSV files from Zillow's research data:
- Home values (ZHVI)
- Forecasts
- Price tiers (top and bottom)
- Sale and listing prices
- Days-to-pending metrics
- Inventory data
- Zestimate accuracy (web-scraped)
- Values by bedroom count (1-5 bedrooms)

## Usage
1. Ensure all required data files are in the same directory as the script
2. Run the script: `python zillow_data_processor.py`
3. The output will be saved as `Zillow_Data.csv`

## Missing Data Handling
The script handles missing data using a hierarchical approach:
1. For missing metro values: Uses state values where available
2. For missing state values: Uses averages of metro values in that state
3. For missing national values: Uses averages of all state values

For details, see the [MISSING_DATA_APPROACH.md](MISSING_DATA_APPROACH.md) file.

## Output File
The `Zillow_Data.csv` file contains columns as specified in the requirements, including:
- Region information (key_row, regiontype, regionname, statename)
- Home values (current, historical, by bedroom count)
- Price tiers
- Sales and listing prices
- Days to pending metrics
- Inventory data
- Zestimate accuracy metrics
