#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Zillow Data Organization Project
Author: Jesse Antony
Date: April 2025

This script processes various Zillow data files and creates a comprehensive CSV 
with home values, forecasts, and other real estate metrics for regions across the US.
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime
import re
import os
import logging
import traceback

def setup_logging():
    """Set up logging configuration with file and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("zillow_processing.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def main():
    """Main function to orchestrate the Zillow data processing."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Zillow Data Organization Project...")
    
    # Load Keys file
    keys_df = load_keys_file(logger)
    
    # Initialize output DataFrame with basic region information
    output_df = initialize_output_df(keys_df, logger)
    
    # Process each data source and add to output
    output_df = process_home_values(output_df, logger)
    output_df = process_forecast_data(output_df, logger)
    output_df = process_price_tiers(output_df, logger)
    output_df = process_sale_listing_prices(output_df, logger)
    output_df = process_days_to_pending(output_df, logger)
    output_df = process_inventory(output_df, logger)
    output_df = process_zestimate_data(output_df, logger)
    output_df = process_bedroom_values(output_df, logger)

   
    # Handle missing data with fallback logic
    output_df = handle_missing_data(output_df, logger)
    # validate_output(output_df)
    # Save the final output

    numeric_columns = [
    'home_value', 'home_value_rounded', 'home_value_change_mm', 'home_value_change_yy',
    'forecastyoypctchange', 'top_tier', 'bottom_tier', 'sale_price', 'listing_price',
    'current_days_to_pending', 'days_to_pending', 'inventory', 
    'zillow_on_market_median_error', 'zillow_on_market_properties', 'zillow_on_market_within_5', 
    'zillow_on_market_within_10', 'zillow_on_market_within_20',
    'zillow_off_market_median_error', 'zillow_off_market_properties', 'zillow_off_market_within_5', 
    'zillow_off_market_within_10', 'zillow_off_market_within_20',
    'home_value_one_bed', 'home_value_two_bed', 'home_value_three_bed', 'home_value_four_bed', 'home_value_five_bed'
    ]

    # Plus yearly median columns
    for year in range(2016, 2026):
        numeric_columns.append(f'{year}_median_home_value')
    
    # Convert all numeric columns to proper numeric format
    logger.info("Converting columns to numeric format...")
    for col in numeric_columns:
        if col in output_df.columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
    
    # Save the final output with proper numeric formatting
    output_df.to_csv('Zillow_Data.csv', index=False, float_format='%.2f')
    logger.info("Zillow_Data.csv has been generated successfully.")
    
    # Create documentation for missing data handling
    document_missing_data_approach(logger)
    
    # Create other supporting files for the repository
    create_readme_file(logger)
    create_requirements_file(logger)
    
    logger.info("Zillow Data Organization Project completed successfully.")

def document_missing_data_approach(logger):
    """Creates documentation explaining the missing data handling approach."""
    logger.info("Creating missing data documentation...")
    with open("MISSING_DATA_APPROACH.md", "w") as f:
        f.write("# Missing Data Handling Approach\n\n")
        f.write("As requested, missing data points were filled using the following hierarchy:\n\n")
        f.write("1. **Missing Metro Values**: Used corresponding state value when available\n")
        f.write("2. **Missing State Values**: Used average of all metro values in that state\n")
        f.write("3. **Missing National Values**: Used average of all state values\n\n")
        f.write("## Implementation Details\n\n")
        f.write("The implementation can be found in the `handle_missing_data()` function, which:\n\n")
        f.write("- Processes each column independently\n")
        f.write("- Determines appropriate data type (text vs. numeric) for each column\n")
        f.write("- For numeric columns, uses mean for aggregation\n")
        f.write("- For text columns, uses mode (most common value) for aggregation\n")
        f.write("- Special attention is given to ensure the United States row has complete data\n")
    logger.info("Missing data documentation created successfully.")

def process_zestimate_data(output_df, logger):
    """
    Scrapes Zestimate accuracy data with state matching and metro disambiguation.
    Handles edge cases like Washington state and New York while avoiding hard-coding.
    
    Args:
        output_df (pd.DataFrame): DataFrame with region data
        logger: Logger instance
    
    Returns:
        pd.DataFrame: DataFrame with added Zestimate columns
    """
    logger.info("Processing Zestimate data...")
    
    # Make a safety copy of the original dataframe
    original_df = output_df.copy()

    try:
        # Standard imports
        import time
        import re
        import random
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        import json
        import datetime
        import traceback
        
        # Install required packages if not already installed
        try:
            import undetected_chromedriver as uc
            from selenium_stealth import stealth
            logger.info("Required packages (undetected-chromedriver, selenium-stealth) found.")
        except ImportError:
            import subprocess
            import sys
            logger.warning("Installing required packages: undetected-chromedriver, selenium-stealth...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                       "undetected-chromedriver", "selenium-stealth"])
                import undetected_chromedriver as uc
                from selenium_stealth import stealth
                logger.info("Packages installed successfully.")
            except Exception as e:
                logger.error(f"Failed to install required packages: {e}")
                logger.error("Please install 'undetected-chromedriver' and 'selenium-stealth' manually.")
                return original_df 
                
        # Selenium imports
        from selenium.webdriver.chrome.options import Options as ChromeOptions 
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

        # --- Helper Functions ---
        def clean_city_name(name):
            """Clean a city/region name for standardized matching."""
            if not isinstance(name, str): return ""
            # Remove common metro suffixes
            name = re.sub(r'-Long Beach-Anaheim', '', name, flags=re.IGNORECASE)
            name = re.sub(r'-Fort Lauderdale', '', name, flags=re.IGNORECASE)
            name = re.sub(r'-St\.? Paul', '', name, flags=re.IGNORECASE) 
            name = re.sub(r'-Fort Worth', '', name, flags=re.IGNORECASE)
            name = re.sub(r'-Newark', '', name, flags=re.IGNORECASE) 
            name = re.sub(r'-Naperville-Elgin', '', name, flags=re.IGNORECASE) 
            # Remove state designations and extra characters
            name = name.split(',')[0].strip()
            name = re.sub(r'\([^)]*\)', '', name).strip()
            name = re.sub(r'[^\w\s]', ' ', name)
            name = re.sub(r'\s+', ' ', name).strip()
            return name.lower()
            
        def normalize_name(name):
            """Aggressively normalize a name for lookup purposes."""
            if not isinstance(name, str): return ""
            # Convert to lowercase, remove all whitespace and punctuation
            return re.sub(r'[^\w]', '', name.lower())

        def parse_number(text):
            """Parse number values from text, handling K/M suffixes."""
            if not isinstance(text, str) or text.lower() in ['na', 'n/a', '-', '', 'not available']: return None
            text = text.replace('%', '').replace(',', '').replace('$', '').strip()
            if 'k' in text.lower():
                try: return float(text.lower().replace('k', '').strip()) * 1000
                except ValueError: logger.warning(f"Failed to parse K notation: {text}"); return None
            elif 'm' in text.lower():
                try: return float(text.lower().replace('m', '').strip()) * 1000000
                except ValueError: logger.warning(f"Failed to parse M notation: {text}"); return None
            else:
                try: return float(text)
                except ValueError: return None
                
        def find_matching_brace(text, start_index):
            """Find matching closing brace in JSON string, handling nested structures."""
            if start_index < 0 or start_index >= len(text) or text[start_index] != '{': 
                return -1 
            
            brace_level = 1
            current_index = start_index + 1
            
            while current_index < len(text):
                char = text[current_index]
                if char == '{': 
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0: 
                        return current_index 
                elif char == '"' or char == "'":
                    quote_char = char
                    current_index += 1
                    while current_index < len(text):
                        if text[current_index] == quote_char and text[current_index-1] != '\\': 
                            break
                        current_index += 1
                current_index += 1
                
            return -1 

        # --- ENHANCED INITIAL SETUP & LOOKUPS ---
        logger.info("Preparing region lookups and detecting ambiguities...")
        
        # Ensure clean region names are available
        if 'clean_regionname' not in output_df.columns:
            logger.info("Generating cleaned region names...")
            output_df['clean_regionname'] = output_df['regionname'].apply(clean_city_name)
        
        # Create multiple lookup approaches for better matching
        state_lookup = {}          # Exact name → index
        state_lookup_norm = {}     # Normalized name → index
        state_lookup_lower = {}    # Lowercase name → index
        state_name_by_index = {}   # Index → original name
        
        metro_lookup = {}          # Exact name → index
        metro_by_clean_name = defaultdict(list)  # Clean name → list of indices
        
        # Log all state names found
        all_states = []
        
        # Build improved state lookups
        logger.info("Building enhanced state lookups...")
        for idx, row in output_df[output_df['regiontype'] == 'state'].iterrows():
            if pd.isna(row['regionname']):
                continue
                
            state_name = row['regionname']
            all_states.append(state_name)
            
            # Store in state lookups
            state_lookup[state_name] = idx
            state_lookup_lower[state_name.lower()] = idx
            state_lookup_norm[normalize_name(state_name)] = idx
            state_name_by_index[idx] = state_name
        
        logger.info(f"Found {len(all_states)} states: {', '.join(all_states)}")
        
        # Verify Washington state
        washington_in_states = any('washington' in state.lower() for state in all_states)
        logger.info(f"Washington state found in states list: {washington_in_states}")
        
        # Find Washington state directly
        wa_indices = [idx for idx, name in state_name_by_index.items() 
                     if 'washington' in name.lower() and 'dc' not in name.lower()]
        if wa_indices:
            wa_idx = wa_indices[0]
            wa_name = state_name_by_index[wa_idx]
            logger.info(f"Found Washington state at index {wa_idx}: '{wa_name}'")
        else:
            logger.warning("Could not find Washington state directly in DataFrame!")
            
        # Find New York state directly
        ny_indices = [idx for idx, name in state_name_by_index.items() 
                     if 'new york' in name.lower()]
        if ny_indices:
            ny_idx = ny_indices[0]
            ny_name = state_name_by_index[ny_idx]
            logger.info(f"Found New York state at index {ny_idx}: '{ny_name}'")
        else:
            logger.warning("Could not find New York state directly in DataFrame!")
        
        # Build metro lookups
        for idx, row in output_df[output_df['regiontype'] == 'metro'].iterrows():
            if pd.isna(row['regionname']):
                continue
                
            metro_name = row['regionname']
            
            # Store in metro lookup
            metro_lookup[metro_name] = idx
            
            # Group by clean name for disambiguation
            clean_name = clean_city_name(metro_name)
            if clean_name:
                metro_by_clean_name[clean_name].append(idx)
        
        # Identify ambiguous metros (multiple matches for same clean name)
        ambiguous_metros = {name: indices for name, indices in metro_by_clean_name.items() 
                           if len(indices) > 1}
        
        logger.info(f"Found {len(ambiguous_metros)} ambiguous metro names with multiple matches.")
        
        # Check New York specifically
        if 'new york' in metro_by_clean_name:
            ny_metro_indices = metro_by_clean_name['new york']
            logger.info(f"Found {len(ny_metro_indices)} metros with 'New York' in the name:")
            for idx in ny_metro_indices:
                metro_name = output_df.loc[idx, 'regionname']
                state_code = output_df.loc[idx, 'statename'] if 'statename' in output_df.columns else 'N/A'
                logger.info(f"  Index {idx}: '{metro_name}' (State: {state_code})")
        
        # Create disambiguation map
        disambiguation_map = {}
        
        for clean_name, indices in ambiguous_metros.items():
            # Group by state for disambiguation
            by_state = defaultdict(list)
            
            for idx in indices:
                # First check if statename column has the state
                if 'statename' in output_df.columns and pd.notna(output_df.loc[idx, 'statename']):
                    state_code = output_df.loc[idx, 'statename']
                    by_state[state_code].append(idx)
                # Then try to extract state from region name
                elif isinstance(output_df.loc[idx, 'regionname'], str) and ',' in output_df.loc[idx, 'regionname']:
                    parts = output_df.loc[idx, 'regionname'].split(',')
                    if len(parts) == 2:
                        potential_code = parts[1].strip()
                        if len(potential_code) == 2:
                            by_state[potential_code.upper()].append(idx)
                            continue
                # If no state found, put in unknown
                by_state['unknown'].append(idx)
            
            # Choose preferred metro for each state
            disambiguation_map[clean_name] = {}
            
            for state_code, state_indices in by_state.items():
                if len(state_indices) == 1:
                    # Only one metro with this name in this state - unambiguous
                    primary_idx = state_indices[0]
                else:
                    # Multiple metros with same name in same state - use longest name as primary
                    name_lengths = {idx: len(str(output_df.loc[idx, 'regionname'])) for idx in state_indices}
                    sorted_indices = sorted(name_lengths.items(), key=lambda x: x[1], reverse=True)
                    primary_idx = sorted_indices[0][0]
                
                disambiguation_map[clean_name][state_code] = primary_idx
        
        # Find US index
        us_indices = output_df[output_df['regionname'] == 'United States'].index
        us_index = us_indices[0] if not us_indices.empty else None
        
        # Log lookup status
        logger.info(f"Built lookups for {len(state_lookup)} states and {len(metro_lookup)} metros.")
        logger.info(f"Created disambiguation map for {len(disambiguation_map)} ambiguous names.")
        
        # Initialize date and URL
        zestimate_date = datetime.datetime.now().strftime('%B %d, %Y')
        zillow_url = "https://www.zillow.com/z/zestimate/"
        
        # --- Selenium Driver Setup ---
        logger.info("Setting up web driver...")
        user_agents = [ 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15"
        ]
        chosen_agent = random.choice(user_agents)
        options = uc.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument(f"--user-agent={chosen_agent}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
            
        driver = None 
        on_market_data = {}
        off_market_data = {}
        page_loaded_correctly = False
        
        # --- Main Scraping Block ---
        try:
            logger.info("Initializing undetected ChromeDriver...")
            driver = uc.Chrome(options=options) 
            logger.info("WebDriver initialized successfully.")
            logger.info("Applying stealth settings...")
            stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", 
                   webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", 
                   fix_hairline=True, run_on_insecure_origins=False)
            logger.info("Stealth applied.")
            
            def random_wait(min_seconds=3, max_seconds=6): 
                time.sleep(random.uniform(min_seconds, max_seconds))
            
            logger.info(f"Navigating to {zillow_url}...")
            driver.get(zillow_url)
            random_wait(12, 18) 

            # --- Verify page loaded correctly ---
            try:
                logger.info("Verifying page content...")
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, "//h2[contains(text(), 'Active listings accuracy')]"))
                )
                logger.info("Verified page content successfully.")
                page_loaded_correctly = True
            except TimeoutException:
                logger.error("Expected content not found on page. Likely blocked or incorrect page.")
                page_title = driver.title
                logger.error(f"Page title: {page_title}")
                page_loaded_correctly = False
            
            # --- Extract last update date if possible ---
            if page_loaded_correctly:
                try:
                    date_element = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//p[contains(., 'Last updated:')]"))
                    )
                    date_text = date_element.text
                    date_match = re.search(r'Last updated:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', date_text)
                    if date_match:
                        zestimate_date = date_match.group(1).strip()
                        logger.info(f"Found last updated date: {zestimate_date}")
                except Exception as e:
                    logger.warning(f"Could not extract date: {e}")

            # --- Extract data only if page loaded correctly ---
            if page_loaded_correctly:
                # ON-MARKET DATA EXTRACTION
                logger.info("--- Extracting On-Market Data ---")
                data_script = driver.execute_script(
                    "var ts=null, p='activeData ='; " +
                    "for (var i=0;i<document.scripts.length;i++){" +
                    "  var sc=document.scripts[i].text; " +
                    "  if(sc.includes(p)||sc.includes('var '+p)){ts=sc; break;}" +
                    "} return ts;"
                )
                
                if data_script:
                    start_pattern = r'(?:var\s+)?activeData\s*=\s*\{'
                    match_start = re.search(start_pattern, data_script)
                    if match_start:
                        start_brace_index = match_start.end() - 1
                        end_brace_index = find_matching_brace(data_script, start_brace_index)
                        
                        if end_brace_index != -1:
                            json_str = data_script[start_brace_index : end_brace_index + 1]
                            try:
                                # Clean any HTML tags in the JSON
                                def remove_html_tags(m): 
                                    return f'"headers": {re.sub(r"(?<!\\)([\"'])(.*?)(<[^>]+>)(.*?)(\1)", r"\1\2\4\5", m.group(1))}'
                                json_str_cleaned = re.sub(r'"headers":\s*(\[[\s\S]*?\])', remove_html_tags, json_str)
                                json_str_cleaned = re.sub(r',\s*([\}\]])', r'\1', json_str_cleaned)
                                
                                zestimate_data_js = json.loads(json_str_cleaned)
                                logger.info("Parsed On-Market JSON successfully.")
                                
                                # Process National data
                                if "National" in zestimate_data_js and "data" in zestimate_data_js["National"]:
                                    for item in zestimate_data_js["National"]["data"]:
                                        if "Region" in item and ("United States" in item["Region"] or "All 50 States" in item["Region"]):
                                            on_market_data["United States"] = {
                                                "category": "national",
                                                "data": {k: parse_number(item.get(v)) for k, v in {
                                                    "median_error": "Median Error", 
                                                    "homes": "Homes With Zestimates", 
                                                    "within_5": "Within 5% of Sale Price", 
                                                    "within_10": "Within 10% of Sale Price", 
                                                    "within_20": "Within 20% of Sale Price"
                                                }.items()}
                                            }
                                            logger.info(" -> Extracted National on-market data")
                                
                                # Process State data
                                if "States" in zestimate_data_js and "data" in zestimate_data_js["States"]:
                                    count = 0
                                    for item in zestimate_data_js["States"]["data"]:
                                        if "State" in item:
                                            state_name = item["State"]
                                            data_values = {k: parse_number(item.get(v)) for k, v in {
                                                "median_error": "Median Error", 
                                                "homes": "Homes With Zestimates", 
                                                "within_5": "Within 5% of Sale Price", 
                                                "within_10": "Within 10% of Sale Price", 
                                                "within_20": "Within 20% of Sale Price"
                                            }.items()}
                                            
                                            on_market_data[state_name] = {
                                                "category": "state",
                                                "data": data_values
                                            }
                                            count += 1
                                            
                                            # For Washington state specifically, store a normalized key too
                                            if state_name.lower() == "washington":
                                                logger.info(f" -> Washington state on-market data: {data_values}")
                                                # Also store with normalized key for easier lookup
                                                on_market_data["washington_state"] = {
                                                    "category": "state",
                                                    "data": data_values
                                                }
                                                
                                            # For New York state, special handling (similar to Washington)
                                            if state_name.lower() == "new york":
                                                logger.info(f" -> New York state on-market data: {data_values}")
                                                # Also store with normalized key for easier lookup
                                                on_market_data["new_york_state"] = {
                                                    "category": "state",
                                                    "data": data_values
                                                }
                                    logger.info(f" -> Extracted {count} states on-market data")
                                
                                # Process Metro data
                                if "Top Metro Areas" in zestimate_data_js and "data" in zestimate_data_js["Top Metro Areas"]:
                                    count = 0
                                    for item in zestimate_data_js["Top Metro Areas"]["data"]:
                                        if "Metropolitan Area" in item:
                                            metro_name = item["Metropolitan Area"]
                                            
                                            # Extract any potential state info from the metro name
                                            state_info = None
                                            if ',' in metro_name and len(metro_name.split(',')) == 2:
                                                state_part = metro_name.split(',')[1].strip()
                                                if len(state_part) == 2:  # Likely a state code
                                                    state_info = state_part
                                            
                                            # Special handling for New York metro (log for debugging)
                                            if "new york" in metro_name.lower():
                                                logger.info(f" -> Found New York metro: '{metro_name}' with state info: {state_info}")
                                            
                                            on_market_data[metro_name] = {
                                                "category": "metro",
                                                "state": state_info,  # Store state info if found
                                                "data": {k: parse_number(item.get(v)) for k, v in {
                                                    "median_error": "Median Error", 
                                                    "homes": "Homes With Zestimates", 
                                                    "within_5": "Within 5% of Sale Price", 
                                                    "within_10": "Within 10% of Sale Price", 
                                                    "within_20": "Within 20% of Sale Price"
                                                }.items()}
                                            }
                                            count += 1
                                    logger.info(f" -> Extracted {count} metros on-market data")
                            except Exception as e:
                                logger.error(f"ERROR processing on-market JSON: {e}")
                                logger.error(traceback.format_exc())
                
                # OFF-MARKET DATA EXTRACTION
                logger.info("--- Extracting Off-Market Data ---")
                off_script = driver.execute_script(
                    "var ts=null, p1='var offMarketData =', p2='offMarketData ='; " +
                    "for(var i=0;i<document.scripts.length;i++){" +
                    "  var sc=document.scripts[i].text; " +
                    "  if(sc.includes(p1)||sc.includes(p2)){ts=sc; break;}" +
                    "} return ts;"
                )
                
                if off_script:
                    start_pattern = r'(?:var\s+)?offMarketData\s*=\s*\{'
                    match_start = re.search(start_pattern, off_script)
                    if match_start:
                        start_brace_index = match_start.end() - 1
                        end_brace_index = find_matching_brace(off_script, start_brace_index)
                        
                        if end_brace_index != -1:
                            json_str = off_script[start_brace_index : end_brace_index + 1]
                            try:
                                # Clean any HTML tags in the JSON
                                def remove_html_tags(m): 
                                    return f'"headers": {re.sub(r"(?<!\\)([\"'])(.*?)(<[^>]+>)(.*?)(\1)", r"\1\2\4\5", m.group(1))}'
                                json_str_cleaned = re.sub(r'"headers":\s*(\[[\s\S]*?\])', remove_html_tags, json_str)
                                json_str_cleaned = re.sub(r',\s*([\}\]])', r'\1', json_str_cleaned)
                                
                                off_market_data_js = json.loads(json_str_cleaned)
                                logger.info("Parsed Off-Market JSON successfully.")
                                
                                # Process National data
                                if "National" in off_market_data_js and "data" in off_market_data_js["National"]:
                                    for item in off_market_data_js["National"]["data"]:
                                        if "Region" in item and ("United States" in item["Region"] or "All 50 States" in item["Region"]):
                                            off_market_data["United States"] = {
                                                "category": "national",
                                                "data": {k: parse_number(item.get(v)) for k, v in {
                                                    "median_error": "Median Error", 
                                                    "homes": "Homes With Zestimates", 
                                                    "within_5": "Within 5% of Sale Price", 
                                                    "within_10": "Within 10% of Sale Price", 
                                                    "within_20": "Within 20% of Sale Price"
                                                }.items()}
                                            }
                                            logger.info(" -> Extracted National off-market data")
                                
                                # Process State data
                                if "States" in off_market_data_js and "data" in off_market_data_js["States"]:
                                    count = 0
                                    for item in off_market_data_js["States"]["data"]:
                                        if "State" in item:
                                            state_name = item["State"]
                                            data_values = {k: parse_number(item.get(v)) for k, v in {
                                                "median_error": "Median Error", 
                                                "homes": "Homes With Zestimates", 
                                                "within_5": "Within 5% of Sale Price", 
                                                "within_10": "Within 10% of Sale Price", 
                                                "within_20": "Within 20% of Sale Price"
                                            }.items()}
                                            
                                            off_market_data[state_name] = {
                                                "category": "state",
                                                "data": data_values
                                            }
                                            count += 1
                                            
                                            # For Washington state specifically, store a normalized key too
                                            if state_name.lower() == "washington":
                                                logger.info(f" -> Washington state off-market data: {data_values}")
                                                # Also store with normalized key for easier lookup
                                                off_market_data["washington_state"] = {
                                                    "category": "state",
                                                    "data": data_values
                                                }
                                                
                                            # For New York state, special handling (similar to Washington)
                                            if state_name.lower() == "new york":
                                                logger.info(f" -> New York state off-market data: {data_values}")
                                                # Also store with normalized key for easier lookup
                                                off_market_data["new_york_state"] = {
                                                    "category": "state",
                                                    "data": data_values
                                                }
                                    logger.info(f" -> Extracted {count} states off-market data")
                                
                                # Process Metro data
                                if "Top Metro Areas" in off_market_data_js and "data" in off_market_data_js["Top Metro Areas"]:
                                    count = 0
                                    for item in off_market_data_js["Top Metro Areas"]["data"]:
                                        if "Metropolitan Area" in item:
                                            metro_name = item["Metropolitan Area"]
                                            
                                            # Extract any potential state info from the metro name
                                            state_info = None
                                            if ',' in metro_name and len(metro_name.split(',')) == 2:
                                                state_part = metro_name.split(',')[1].strip()
                                                if len(state_part) == 2:  # Likely a state code
                                                    state_info = state_part
                                                    
                                            # Special handling for New York metro (log for debugging)
                                            if "new york" in metro_name.lower():
                                                logger.info(f" -> Found New York metro: '{metro_name}' with state info: {state_info}")
                                            
                                            off_market_data[metro_name] = {
                                                "category": "metro",
                                                "state": state_info,  # Store state info if found
                                                "data": {k: parse_number(item.get(v)) for k, v in {
                                                    "median_error": "Median Error", 
                                                    "homes": "Homes With Zestimates", 
                                                    "within_5": "Within 5% of Sale Price", 
                                                    "within_10": "Within 10% of Sale Price", 
                                                    "within_20": "Within 20% of Sale Price"
                                                }.items()}
                                            }
                                            count += 1
                                    logger.info(f" -> Extracted {count} metros off-market data")
                            except Exception as e:
                                logger.error(f"ERROR processing off-market JSON: {e}")
                                logger.error(traceback.format_exc())
                
                # Summarize extraction results
                logger.info("--- DATA EXTRACTION SUMMARY ---")
                on_market_states = sum(1 for v in on_market_data.values() if isinstance(v, dict) and v.get("category") == "state")
                on_market_metros = sum(1 for v in on_market_data.values() if isinstance(v, dict) and v.get("category") == "metro")
                off_market_states = sum(1 for v in off_market_data.values() if isinstance(v, dict) and v.get("category") == "state")
                off_market_metros = sum(1 for v in off_market_data.values() if isinstance(v, dict) and v.get("category") == "metro")
                
                logger.info(f"On-market data: {len(on_market_data)} total ({on_market_states} states, {on_market_metros} metros)")
                logger.info(f"Off-market data: {len(off_market_data)} total ({off_market_states} states, {off_market_metros} metros)")
            else:
                logger.warning("Skipping data extraction due to page load issues.")
                
        except Exception as e:
            logger.error(f"ERROR during scraping: {e}")
            logger.error(traceback.format_exc())
        finally:
            if driver:
                logger.info("Closing the browser...")
                try: 
                    driver.quit()
                    logger.info("Browser closed.")
                except Exception as e: 
                    logger.warning(f"Error closing browser: {e}")

        # --- Apply Scraped Data to DataFrame ---
        logger.info("--- APPLYING SCRAPED DATA TO DATAFRAME ---")
        output_df['zestimate_accuracy'] = zestimate_date
        zestimate_columns = [
            'zillow_on_market_median_error', 'zillow_on_market_properties', 'zillow_on_market_within_5', 
            'zillow_on_market_within_10', 'zillow_on_market_within_20',
            'zillow_off_market_median_error', 'zillow_off_market_properties', 'zillow_off_market_within_5', 
            'zillow_off_market_within_10', 'zillow_off_market_within_20'
        ]
        for col in zestimate_columns:
            if col not in output_df.columns: 
                output_df[col] = np.nan
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce')

        # --- Enhanced Data Application Function ---
        def apply_data_with_enhanced_matching(market_data, data_type="on-market"):
            """
            Apply data with improved matching:
            1. Multiple state lookup methods
            2. Direct row finding fallback
            3. Systematic metro disambiguation
            4. Strict type checking to prevent confusion
            """
            processed_indices = set()
            match_count = 0
            
            # Column mapping for zestimate data
            prefix = f"zillow_{data_type.replace('-', '_')}"
            col_map = {
                "median_error": "_median_error", 
                "homes": "_properties", 
                "within_5": "_within_5", 
                "within_10": "_within_10", 
                "within_20": "_within_20"
            }
            
            # New York and Washington specific tracking
            ny_state_data_applied = False
            wa_state_data_applied = False
            ny_metros_data_applied = []
            
            # Apply data to one or more indices
            def apply_to_indices(indices, data_dict, source, required_type=None):
                nonlocal processed_indices, match_count, ny_state_data_applied, wa_state_data_applied, ny_metros_data_applied
                if not indices or not isinstance(data_dict, dict):
                    return False
                
                applied = False
                for idx in indices:
                    if idx in processed_indices:
                        logger.debug(f"  Already processed index {idx}, skipping")
                        continue
                        
                    if idx not in output_df.index:
                        logger.warning(f"  Warning: Index {idx} not found in DataFrame")
                        continue
                    
                    # Get the row's actual region type and name
                    actual_type = output_df.loc[idx, 'regiontype'] if pd.notna(output_df.loc[idx, 'regiontype']) else None
                    region_name = output_df.loc[idx, 'regionname'] if pd.notna(output_df.loc[idx, 'regionname']) else "Unknown"
                    
                    # STRICT TYPE CHECKING - prevents applying state data to metro rows and vice versa
                    if required_type and actual_type != required_type:
                        logger.warning(f" BLOCKED: Cannot apply {required_type} data to {actual_type} row for '{region_name}'")
                        continue
                    
                    # Track New York specific applications
                    is_ny_state = (actual_type == 'state' and region_name.lower() == 'new york')
                    is_ny_metro = (actual_type == 'metro' and 'new york' in region_name.lower())
                    
                    # Track Washington specific applications
                    is_wa_state = (actual_type == 'state' and region_name.lower() == 'washington')
                    
                    if is_ny_state:
                        logger.info(f" Applying to NEW YORK STATE: {region_name}")
                        old_values = {}
                        for col_key, col_suffix in col_map.items():
                            col_name = prefix + col_suffix
                            old_values[col_name] = output_df.loc[idx, col_name] if col_name in output_df.columns else None
                        logger.info(f"  Before values: {old_values}")
                        ny_state_data_applied = True
                    elif is_ny_metro:
                        logger.info(f" Applying to NEW YORK METRO: {region_name}")
                        old_values = {}
                        for col_key, col_suffix in col_map.items():
                            col_name = prefix + col_suffix
                            old_values[col_name] = output_df.loc[idx, col_name] if col_name in output_df.columns else None
                        logger.info(f"  Before values: {old_values}")
                        ny_metros_data_applied.append(region_name)
                    elif is_wa_state:
                        logger.info(f" Applying to WASHINGTON STATE: {region_name}")
                        old_values = {}
                        for col_key, col_suffix in col_map.items():
                            col_name = prefix + col_suffix
                            old_values[col_name] = output_df.loc[idx, col_name] if col_name in output_df.columns else None
                        logger.info(f"  Before values: {old_values}")
                        wa_state_data_applied = True
                    
                    # Apply each value to the appropriate column
                    for col_key, col_suffix in col_map.items():
                        value = data_dict.get(col_key)
                        if value is not None:
                            col_name = prefix + col_suffix
                            try:
                                output_df.loc[idx, col_name] = value
                                applied = True
                            except Exception as e:
                                logger.error(f"  Error applying {col_key} to {idx}: {e}")
                    
                    if applied:
                        processed_indices.add(idx)
                        match_count += 1
                        logger.info(f" Applied {data_type} data from '{source}' to {actual_type} '{region_name}'")
                        
                        # Log after values for New York or Washington
                        if is_ny_state or is_ny_metro or is_wa_state:
                            after_values = {}
                            for col_key, col_suffix in col_map.items():
                                col_name = prefix + col_suffix
                                after_values[col_name] = output_df.loc[idx, col_name] if col_name in output_df.columns else None
                            logger.info(f"  After values: {after_values}")
                
                return applied
            
            # STEP 1: Apply National data
            if "United States" in market_data and us_index is not None:
                us_info = market_data["United States"]
                if isinstance(us_info, dict) and "category" in us_info and "data" in us_info:
                    apply_to_indices([us_index], us_info["data"], "United States", "country")
            
            # STEP 2: Apply State data with multiple lookup approaches
            logger.info(f"\nApplying {data_type} STATE data...")
            for region_name, region_info in market_data.items():
                if not isinstance(region_info, dict) or region_info.get("category") != "state":
                    continue
                
                # Skip the special washington_state and new_york_state entries (they're duplicates)
                if region_name in ["washington_state", "new_york_state"]:
                    continue
                    
                state_data = region_info.get("data", {})
                applied = False
                
                # Method 1: Exact match in state_lookup
                if region_name in state_lookup:
                    logger.info(f"  Found exact match in state_lookup for '{region_name}'")
                    if apply_to_indices([state_lookup[region_name]], state_data, region_name, "state"):
                        applied = True
                        continue
                
                # Method 2: Case-insensitive match
                if not applied and region_name.lower() in state_lookup_lower:
                    logger.info(f"  Found case-insensitive match in state_lookup for '{region_name}'")
                    if apply_to_indices([state_lookup_lower[region_name.lower()]], state_data, region_name, "state"):
                        applied = True
                        continue
                
                # Method 3: Normalized match
                if not applied:
                    norm_name = normalize_name(region_name)
                    if norm_name in state_lookup_norm:
                        logger.info(f"  Found normalized match in state_lookup for '{region_name}'")
                        if apply_to_indices([state_lookup_norm[norm_name]], state_data, region_name, "state"):
                            applied = True
                            continue
                
                # Method 4: Direct DataFrame filtering (most robust but slower)
                if not applied:
                    # Find state rows that match this name (case-insensitive)
                    matching_states = output_df[
                        (output_df['regiontype'] == 'state') & 
                        (output_df['regionname'].str.lower() == region_name.lower())
                    ]
                    
                    if not matching_states.empty:
                        logger.info(f"  Found direct DataFrame match for state '{region_name}'")
                        state_indices = matching_states.index.tolist()
                        if apply_to_indices(state_indices, state_data, f"{region_name} (direct match)", "state"):
                            applied = True
                            continue
                
                if not applied:
                    logger.warning(f"  No matching state found for '{region_name}'")
            
            # Special check for Washington state - if it hasn't been applied yet
            if not wa_state_data_applied and "washington_state" in market_data:
                logger.info("\n SPECIAL HANDLING FOR WASHINGTON STATE")
                wa_state_rows = output_df[
                    (output_df['regiontype'] == 'state') & 
                    (output_df['regionname'].str.lower() == 'washington')
                ]
                
                if not wa_state_rows.empty:
                    wa_idx = wa_state_rows.index[0]
                    wa_data = market_data["washington_state"].get("data", {})
                    if apply_to_indices([wa_idx], wa_data, "Washington (special handling)", "state"):
                        logger.info(f"  Successfully applied Washington state data via special handling")
            
            # Special check for New York state - if it hasn't been applied yet
            if not ny_state_data_applied and "new_york_state" in market_data:
                logger.info("\n SPECIAL HANDLING FOR NEW YORK STATE")
                ny_state_rows = output_df[
                    (output_df['regiontype'] == 'state') & 
                    (output_df['regionname'].str.lower() == 'new york')
                ]
                
                if not ny_state_rows.empty:
                    ny_idx = ny_state_rows.index[0]
                    ny_data = market_data["new_york_state"].get("data", {})
                    if apply_to_indices([ny_idx], ny_data, "New York (special handling)", "state"):
                        logger.info(f"  Successfully applied New York state data via special handling")
            
            # STEP 3: Apply Metro data with disambiguation
            logger.info(f"\nApplying {data_type} METRO data...")
            for region_name, region_info in market_data.items():
                if not isinstance(region_info, dict) or region_info.get("category") != "metro":
                    continue
                
                metro_data = region_info.get("data", {})
                state_info = region_info.get("state")
                clean_region = clean_city_name(region_name)
                applied = False
                
                # Skip if this is a New York metro and we want special handling later
                if "new york" in region_name.lower() and "NY" == state_info:
                    continue
                
                # Method 1: Direct exact match
                if region_name in metro_lookup:
                    logger.info(f"  Found exact match in metro_lookup for '{region_name}'")
                    idx = metro_lookup[region_name]
                    if apply_to_indices([idx], metro_data, region_name, "metro"):
                        applied = True
                        continue
                
                # Method 2: Clean name matching with disambiguation
                if not applied and clean_region in metro_by_clean_name:
                    matching_indices = metro_by_clean_name[clean_region]
                    logger.info(f"  Found {len(matching_indices)} matches for clean name '{clean_region}'")
                    
                    if len(matching_indices) == 1:
                        # Only one metro with this name - unambiguous
                        idx = matching_indices[0]
                        logger.info(f"  Single match found for '{clean_region}'")
                        if apply_to_indices([idx], metro_data, f"{region_name} (clean match)", "metro"):
                            applied = True
                            continue
                    else:
                        # Multiple metros with this name - disambiguation needed
                        
                        # Method 2a: Filter by state if available
                        if state_info:
                            state_matches = []
                            for idx in matching_indices:
                                if idx in output_df.index and 'statename' in output_df.columns:
                                    df_state = output_df.loc[idx, 'statename']
                                    if pd.notna(df_state) and df_state.upper() == state_info.upper():
                                        state_matches.append(idx)
                            
                            if state_matches:
                                logger.info(f"  Found {len(state_matches)} matches for '{clean_region}' in state '{state_info}'")
                                if len(state_matches) == 1:
                                    # Unique match by state code
                                    if apply_to_indices(state_matches, metro_data, f"{region_name} (state match: {state_info})", "metro"):
                                        applied = True
                                        continue
                                else:
                                    # Multiple matches in same state - use first
                                    if apply_to_indices([state_matches[0]], metro_data, f"{region_name} (first match in {state_info})", "metro"):
                                        applied = True
                                        continue
                        
                        # Method 2b: Use disambiguation map
                        if clean_region in disambiguation_map:
                            logger.info(f"  Using disambiguation map for '{clean_region}'")
                            # Try to match by state first
                            if state_info and state_info in disambiguation_map[clean_region]:
                                primary_idx = disambiguation_map[clean_region][state_info]
                                if apply_to_indices([primary_idx], metro_data, f"{region_name} (primary for {state_info})", "metro"):
                                    applied = True
                                    continue
                            
                            # Then try first available primary
                            logger.info(f"  Trying first available primary from disambiguation map")
                            for state_code, primary_idx in disambiguation_map[clean_region].items():
                                if primary_idx not in processed_indices:
                                    if apply_to_indices([primary_idx], metro_data, f"{region_name} (first available primary)", "metro"):
                                        applied = True
                                        break
                
                if not applied:
                    logger.warning(f"  Could not find suitable match for metro '{region_name}'")
            
            # Special handling for New York metros
            logger.info("\n SPECIAL HANDLING FOR NEW YORK METROS")
            if "new york" in metro_by_clean_name:
                # Find New York metros specifically
                ny_metro_indices = []
                for idx in metro_by_clean_name["new york"]:
                    if idx in output_df.index and 'statename' in output_df.columns:
                        if pd.notna(output_df.loc[idx, 'statename']) and output_df.loc[idx, 'statename'].upper() == 'NY':
                            ny_metro_indices.append(idx)
                
                if ny_metro_indices:
                    logger.info(f"  Found {len(ny_metro_indices)} New York metros")
                    
                    # Find NY metro in the market data
                    ny_metro_data = None
                    for name, info in market_data.items():
                        if isinstance(info, dict) and info.get("category") == "metro" and "new york" in name.lower():
                            if info.get("state") == "NY":
                                logger.info(f"  Found NY metro data for '{name}'")
                                ny_metro_data = info.get("data", {})
                                break
                    
                    if ny_metro_data:
                        # Apply to all NY metros that match
                        for idx in ny_metro_indices:
                            if idx not in processed_indices:
                                metro_name = output_df.loc[idx, 'regionname'] if pd.notna(output_df.loc[idx, 'regionname']) else "Unknown"
                                apply_to_indices([idx], ny_metro_data, f"New York metro (special handling): {metro_name}", "metro")
                    else:
                        logger.warning("  No NY metro data found for special handling")
            
            return match_count
        
        # Apply data with enhanced matching
        on_count = apply_data_with_enhanced_matching(on_market_data, "on-market")
        off_count = apply_data_with_enhanced_matching(off_market_data, "off-market")
        
        logger.info(f"\nApplied on-market data to {on_count} rows")
        logger.info(f"Applied off-market data to {off_count} rows")
        
        # --- Final Verification Step ---
        logger.info("\n--- VERIFICATION CHECKS ---")
        
        # Check New York state data
        ny_state_rows = output_df[
            (output_df['regiontype'] == 'state') & 
            (output_df['regionname'].str.lower() == 'new york')
        ]
        
        if not ny_state_rows.empty:
            ny_state_idx = ny_state_rows.index[0]
            logger.info(f"New York state data:")
            for prefix in ['zillow_on_market', 'zillow_off_market']:
                logger.info(f"  {prefix.replace('_', ' ').upper()}:")
                for suffix in ['_median_error', '_properties', '_within_5', '_within_10', '_within_20']:
                    col = prefix + suffix
                    value = output_df.loc[ny_state_idx, col] if col in output_df.columns else None
                    logger.info(f"    {col} = {value}")
        
        # Check New York metros data
        ny_metro_rows = output_df[
            (output_df['regiontype'] == 'metro') & 
            (output_df['regionname'].str.contains('New York', case=False, na=False))
        ]
        
        if not ny_metro_rows.empty:
            logger.info(f"\nNew York metro data:")
            for idx in ny_metro_rows.index:
                metro_name = output_df.loc[idx, 'regionname']
                logger.info(f"  Metro: {metro_name}")
                for prefix in ['zillow_on_market', 'zillow_off_market']:
                    logger.info(f"    {prefix.replace('_', ' ').upper()}:")
                    for suffix in ['_median_error', '_properties', '_within_5', '_within_10', '_within_20']:
                        col = prefix + suffix
                        value = output_df.loc[idx, col] if col in output_df.columns else None
                        logger.info(f"      {col} = {value}")
        
        # Check for conflicts between NY state and metros
        if not ny_state_rows.empty and not ny_metro_rows.empty:
            logger.info("\nChecking for conflicts between NY state and metros:")
            state_idx = ny_state_rows.index[0]
            
            for idx in ny_metro_rows.index:
                metro_name = output_df.loc[idx, 'regionname']
                conflicts = []
                
                for prefix in ['zillow_on_market', 'zillow_off_market']:
                    for suffix in ['_median_error', '_properties', '_within_5', '_within_10', '_within_20']:
                        col = prefix + suffix
                        if col in output_df.columns:
                            state_val = output_df.loc[state_idx, col]
                            metro_val = output_df.loc[idx, col]
                            
                            if pd.notna(state_val) and pd.notna(metro_val) and state_val == metro_val:
                                conflicts.append(col)
                
                if conflicts:
                    logger.warning(f"  CONFLICT: New York state and metro '{metro_name}' have identical values for: {', '.join(conflicts)}")
                else:
                    logger.info(f"   No conflicts found between New York state and metro '{metro_name}'")
        
        logger.info("\nZestimate data processing complete.")
        return output_df
        
    except Exception as e:
        logger.error(f"ERROR in process_zestimate_data: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("Returning original DataFrame due to error.")
        return original_df

def get_state_abbreviation(state_name):
    """Get two-letter postal abbreviation for a state name."""
    state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
        'District of Columbia': 'DC', 'United States': 'US'
    }
    return state_abbrev.get(state_name, '')

def get_latest_date_columns(df):
    """Get the date columns and latest date from a DataFrame."""
    date_cols = [col for col in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
    date_cols.sort()  # Sort chronologically
    return date_cols, date_cols[-1] if date_cols else None

def create_value_lookup(df, date_col):
    """Create a lookup dictionary mapping RegionName to values for a given date column."""
    if date_col is None:
        return {}
    return dict(zip(df['RegionName'], df[date_col]))

def calculate_yearly_medians(df, region_col='RegionName'):
    """Calculate median values for each year for each region."""
    date_cols = [col for col in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
    years = set(int(col.split('-')[0]) for col in date_cols)
    
    medians = {}
    for year in years:
        year_cols = [col for col in date_cols if col.startswith(str(year))]
        year_medians = {}
        
        for _, row in df.iterrows():
            region = row[region_col]
            values = [row[col] for col in year_cols if pd.notna(row[col])]
            if values:
                year_medians[region] = np.median(values)
        
        medians[year] = year_medians
    
    return medians

def load_keys_file(logger):
    """Load the Keys file and return as DataFrame."""
    logger.info("Loading Keys file...")
    
    try:
        # Use the second row (index 1) as the header
        keys_df = pd.read_csv('Keys - KEYS.csv', header=1)
        
        # Filter rows where RegionName (Column C) is not empty
        keys_df = keys_df[keys_df['RegionName'].notna()]
        
        logger.info(f"Loaded {len(keys_df)} regions from Keys file.")
        return keys_df
    except FileNotFoundError:
        logger.error("Keys file not found! Please ensure 'Keys - KEYS.csv' exists in the current directory.")
        raise
    except Exception as e:
        logger.error(f"Error loading Keys file: {str(e)}")
        raise

def initialize_output_df(keys_df, logger):
    """Initialize the output DataFrame with basic region information."""
    logger.info("Initializing output DataFrame...")
    
    output_df = pd.DataFrame()
    output_df['key_row'] = keys_df['key_row']  # Column A from Keys
    output_df['regiontype'] = keys_df['region_type']  # Column F from Keys
    output_df['regionname'] = keys_df['RegionName']  # Column C from Keys
    
    # Determine statename based on regiontype
    def get_state_name(row):
        if row['regiontype'] == 'country':
            return 'US'
        elif row['regiontype'] == 'state':
            return get_state_abbreviation(row['regionname'])
        elif row['regiontype'] == 'metro':
            # Extract state from regionname (assumes format like "New York, NY")
            match = re.search(r',\s*([A-Z]{2})$', row['regionname'])
            return match.group(1) if match else ''
        return ''
    
    output_df['statename'] = output_df.apply(get_state_name, axis=1)
    logger.info(f"Initialized output DataFrame with {len(output_df)} rows.")
    return output_df

def handle_missing_data(output_df, logger):
    """
    Comprehensive missing data filling following the specified hierarchy.
    """
    logger.info("Applying fallback logic for missing data...")
    
    # Make a copy to avoid modifying the original during iteration
    df = output_df.copy()
    
    # Identify columns by type
    metadata_cols = ['key_row', 'regiontype', 'regionname', 'statename', 'clean_regionname']
    text_cols = ['date', 'dtp_month', 'zestimate_accuracy']  # Non-numeric string columns
    
    # Get all data columns needing processing
    all_data_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Track filled values
    fill_counts = {'metro_from_state': 0, 'state_from_metros': 0, 'national_from_states': 0}
    
    # STEP 1: Process each column independently
    for col in all_data_cols:
        # Skip if no missing values
        if not df[col].isna().any():
            continue
        
        # Determine column type
        is_text = col in text_cols
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) if not is_text else False
        
        # For non-text columns that might be numeric
        if not is_text and not is_numeric:
            try:
                # Try converting to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                is_numeric = True
            except:
                pass
        
        # FILL MISSING METRO VALUES FROM STATE VALUES
        metro_missing = df[(df['regiontype'] == 'metro') & df[col].isna()].index
        for idx in metro_missing:
            state = df.loc[idx, 'statename']
            if pd.isna(state) or state == '':
                continue
            
            # Find state values for this state
            state_values = df[(df['regiontype'] == 'state') & 
                             (df['statename'] == state)][col].dropna()
            
            if not state_values.empty:
                df.loc[idx, col] = state_values.iloc[0]
                fill_counts['metro_from_state'] += 1
        
        # FILL MISSING STATE VALUES FROM METRO AVERAGES
        state_missing = df[(df['regiontype'] == 'state') & df[col].isna()].index
        for idx in state_missing:
            state = df.loc[idx, 'statename']
            if pd.isna(state) or state == '':
                continue
            
            # Find metro values for this state
            metro_values = df[(df['regiontype'] == 'metro') & 
                             (df['statename'] == state)][col].dropna()
            
            if not metro_values.empty:
                if is_text:
                    # Use mode for text columns
                    most_common = metro_values.mode()
                    if not most_common.empty:
                        df.loc[idx, col] = most_common.iloc[0]
                        fill_counts['state_from_metros'] += 1
                else:
                    # Use mean for numeric columns
                    df.loc[idx, col] = metro_values.mean()
                    fill_counts['state_from_metros'] += 1
        
        # FILL MISSING NATIONAL VALUES FROM STATE AVERAGES
        us_missing = df[(df['regionname'] == 'United States') & df[col].isna()].index
        if not us_missing.empty:
            # Find state values (exclude 'United States')
            state_values = df[(df['regiontype'] == 'state')][col].dropna()
            
            if not state_values.empty:
                if is_text:
                    # Use mode for text columns
                    most_common = state_values.mode()
                    if not most_common.empty:
                        df.loc[us_missing, col] = most_common.iloc[0]
                        fill_counts['national_from_states'] += len(us_missing)
                else:
                    # Use mean for numeric columns
                    df.loc[us_missing, col] = state_values.mean()
                    fill_counts['national_from_states'] += len(us_missing)
    
    # STEP 2: Ensure United States has values for ALL metrics
    us_row = df[df['regionname'] == 'United States']
    if not us_row.empty:
        us_idx = us_row.index[0]
        for col in all_data_cols:
            if pd.isna(df.loc[us_idx, col]):
                # Last attempt to fill US data from states
                state_values = df[df['regiontype'] == 'state'][col].dropna()
                if not state_values.empty:
                    if col in text_cols:
                        df.loc[us_idx, col] = state_values.mode().iloc[0]
                    else:
                        df.loc[us_idx, col] = state_values.mean()
                    fill_counts['national_from_states'] += 1
    
    # Summary report
    total_filled = sum(fill_counts.values())
    logger.info(f"Missing data filling complete:")
    logger.info(f"- Filled {fill_counts['metro_from_state']} missing metro values using state values")
    logger.info(f"- Filled {fill_counts['state_from_metros']} missing state values using metro averages")
    logger.info(f"- Filled {fill_counts['national_from_states']} missing national values using state averages")
    logger.info(f"- Total values filled: {total_filled}")
    
    return df

def process_home_values(output_df, logger):
    """
    Process home value data (Metro, State, and US if present in those files)
    and add to output DataFrame.
    """
    logger.info("Processing home values (Metro, State, US)...")

    # --- Load Data (State and Metro only) ---
    try:
        metro_zhvi_df = pd.read_csv('Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
        state_zhvi_df = pd.read_csv('State_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
        # No separate national file loaded, assuming US data is in state/metro files
    except FileNotFoundError as e:
        logger.error(f"Error loading home value file: {e}. Please ensure Metro and State CSV files are present.")
        return output_df

    # --- Find Date Columns and Latest Dates ---
    metro_date_cols, metro_latest_date = get_latest_date_columns(metro_zhvi_df)
    state_date_cols, state_latest_date = get_latest_date_columns(state_zhvi_df)
    logger.info(f"Latest metro date: {metro_latest_date}")
    logger.info(f"Latest state date: {state_latest_date}")

    # --- Create Lookups (Including US if present in source DFs) ---
    metro_home_values = create_value_lookup(metro_zhvi_df, metro_latest_date)
    state_home_values = create_value_lookup(state_zhvi_df, state_latest_date)

    # --- Lookups for Change Calculations ---
    # Month-over-Month
    metro_second_latest_date = metro_date_cols[-2] if len(metro_date_cols) >= 2 else None
    state_second_latest_date = state_date_cols[-2] if len(state_date_cols) >= 2 else None
    metro_prev_month_values = create_value_lookup(metro_zhvi_df, metro_second_latest_date) if metro_second_latest_date else {}
    state_prev_month_values = create_value_lookup(state_zhvi_df, state_second_latest_date) if state_second_latest_date else {}

    # Year-over-Year
    metro_yoy_date = metro_date_cols[-13] if len(metro_date_cols) >= 13 else None
    state_yoy_date = state_date_cols[-13] if len(state_date_cols) >= 13 else None
    metro_yoy_values = create_value_lookup(metro_zhvi_df, metro_yoy_date) if metro_yoy_date else {}
    state_yoy_values = create_value_lookup(state_zhvi_df, state_yoy_date) if state_yoy_date else {}

    # --- Calculate Yearly Medians (Including US if present) ---
    logger.info("Calculating yearly median home values...")
    metro_yearly_medians = calculate_yearly_medians(metro_zhvi_df)
    state_yearly_medians = calculate_yearly_medians(state_zhvi_df)

    # --- Apply Home Values ---
    def get_home_value(row):
        region_name = row['regionname']
        if row['regiontype'] == 'metro':
            return metro_home_values.get(region_name, None)
        elif row['regiontype'] == 'state':
            return state_home_values.get(region_name, None)
        # --- MODIFICATION START: Handle country by checking state/metro lookups ---
        elif row['regiontype'] == 'country':
            # Prefer state lookup for US, fallback to metro
            value = state_home_values.get(region_name, None)
            if value is None:
                value = metro_home_values.get(region_name, None)
            return value
        # --- MODIFICATION END ---
        return None

    output_df['home_value'] = output_df.apply(get_home_value, axis=1)

    # --- Round Home Values ---
    output_df['home_value_rounded'] = output_df['home_value'].apply(
        lambda x: round(x / 1000) * 1000 if pd.notna(x) else None
    )

    # --- Calculate Month-to-Month Change ---
    def get_mm_change(row):
        current_value = row['home_value']
        region_name = row['regionname']
        if pd.isna(current_value):
            return None

        prev_value = None
        if row['regiontype'] == 'metro':
            prev_value = metro_prev_month_values.get(region_name, None)
        elif row['regiontype'] == 'state':
            prev_value = state_prev_month_values.get(region_name, None)
        # --- MODIFICATION START: Handle country ---
        elif row['regiontype'] == 'country':
             # Prefer state lookup for previous month, fallback to metro
            prev_value = state_prev_month_values.get(region_name, None)
            if prev_value is None:
                 prev_value = metro_prev_month_values.get(region_name, None)
        # --- MODIFICATION END ---

        if pd.notna(prev_value) and prev_value != 0:
            return (current_value / prev_value - 1) * 100
        return None

    output_df['home_value_change_mm'] = output_df.apply(get_mm_change, axis=1)

    # --- Calculate Year-over-Year Change ---
    def get_yoy_change(row):
        current_value = row['home_value']
        region_name = row['regionname']
        if pd.isna(current_value):
            return None

        prev_value = None
        if row['regiontype'] == 'metro':
            prev_value = metro_yoy_values.get(region_name, None)
        elif row['regiontype'] == 'state':
            prev_value = state_yoy_values.get(region_name, None)
        # --- MODIFICATION START: Handle country ---
        elif row['regiontype'] == 'country':
             # Prefer state lookup for previous year, fallback to metro
            prev_value = state_yoy_values.get(region_name, None)
            if prev_value is None:
                 prev_value = metro_yoy_values.get(region_name, None)
        # --- MODIFICATION END ---

        if pd.notna(prev_value) and prev_value != 0:
            return (current_value / prev_value - 1) * 100
        return None

    output_df['home_value_change_yy'] = output_df.apply(get_yoy_change, axis=1)

    # --- Add Yearly Median Home Values ---
    logger.info("Adding yearly median home values to output...")
    required_years = range(2016, 2026) # As per requirement doc

    for year in required_years:
        col_name = f'{year}_median_home_value'
        def get_yearly_median(row):
            region_name = row['regionname']
            if row['regiontype'] == 'metro':
                return metro_yearly_medians.get(year, {}).get(region_name, None)
            elif row['regiontype'] == 'state':
                return state_yearly_medians.get(year, {}).get(region_name, None)
            # --- MODIFICATION START: Handle country ---
            elif row['regiontype'] == 'country':
                 # Prefer state median lookup, fallback to metro
                median = state_yearly_medians.get(year, {}).get(region_name, None)
                if median is None:
                     median = metro_yearly_medians.get(year, {}).get(region_name, None)
                return median
            # --- MODIFICATION END ---
            return None

        output_df[col_name] = output_df.apply(get_yearly_median, axis=1)
        logger.info(f"Processed median values for {year}.")

    logger.info("Finished processing home values.")
    return output_df

def process_forecast_data(output_df, logger):
    """Process forecast data and add to output DataFrame."""
    logger.info("Processing forecast data...")

    # Load forecast data - Requirements only specify Metro file
    try:
        forecast_df = pd.read_csv('Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    except FileNotFoundError as e:
        logger.error(f"Error loading forecast file: {e}. Please ensure Metro forecast CSV file is present.")
        # Add columns even if file is missing, fill with NaN
        output_df['date'] = None
        output_df['forecastyoypctchange'] = np.nan
        return output_df

    # Check essential 'BaseDate' column
    if 'BaseDate' not in forecast_df.columns:
        logger.error("'BaseDate' column not found in forecast file!")
        output_df['date'] = None
        output_df['forecastyoypctchange'] = np.nan
        return output_df

    # Find the most recent forecast column (value column)
    forecast_cols = [col for col in forecast_df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
    # Exclude BaseDate if it matches the pattern
    forecast_cols = [col for col in forecast_cols if col != 'BaseDate']
    latest_forecast_col = forecast_cols[-1] if forecast_cols else None

    if latest_forecast_col:
         logger.info(f"Using forecast value column: {latest_forecast_col}")
    else:
        logger.warning("No forecast value column found.")

    # Create lookup dictionaries (will include 'United States' if present in forecast_df)
    base_dates = create_value_lookup(forecast_df, 'BaseDate')
    forecasts = create_value_lookup(forecast_df, latest_forecast_col) if latest_forecast_col else {}

    # Format date helper
    def format_base_date(date_str):
        if pd.isna(date_str): return None
        try:
            # Use str() to handle potential non-string types before parsing
            date_obj = datetime.datetime.strptime(str(date_str), '%Y-%m-%d')
            return date_obj.strftime('%B %d, %Y').replace(' 0', ' ')
        except ValueError: # Catch specific error
            logger.warning(f"Could not format date '{date_str}'")
            return None # Return None if formatting fails

    # Apply Base Date
    def get_base_date(row):
        region_name = row['regionname']
        # MODIFICATION START: Check country also, use existing base_dates lookup
        if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
            date_str = base_dates.get(region_name, None)
            return format_base_date(date_str)
        # MODIFICATION END
        return None

    output_df['date'] = output_df.apply(get_base_date, axis=1)

    # Apply Forecast Value
    if not latest_forecast_col:
         output_df['forecastyoypctchange'] = np.nan # Ensure column exists
    else:
        def get_forecast(row):
            region_name = row['regionname']
            # MODIFICATION START: Check country also, use existing forecasts lookup
            if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
                return forecasts.get(region_name, None)
            # MODIFICATION END
            return None
        output_df['forecastyoypctchange'] = output_df.apply(get_forecast, axis=1)

    logger.info("Finished processing forecast data.")
    return output_df

def process_price_tiers(output_df, logger):
    """Process top and bottom tier home values and add to output DataFrame."""
    logger.info("Processing price tiers...")

    try:
        # Load top tier data
        metro_top_df = pd.read_csv('Metro_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv')
        state_top_df = pd.read_csv('State_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv')
        # Load bottom tier data
        metro_bottom_df = pd.read_csv('Metro_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv')
        state_bottom_df = pd.read_csv('State_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv')
    except FileNotFoundError as e:
        logger.error(f"Error loading price tier file: {e}.")
        output_df['top_tier'] = np.nan
        output_df['bottom_tier'] = np.nan
        return output_df

    # Get latest dates
    _, metro_top_latest = get_latest_date_columns(metro_top_df)
    _, state_top_latest = get_latest_date_columns(state_top_df)
    _, metro_bottom_latest = get_latest_date_columns(metro_bottom_df)
    _, state_bottom_latest = get_latest_date_columns(state_bottom_df)

    # Create lookup dictionaries (includes US if present in source files)
    metro_top_values = create_value_lookup(metro_top_df, metro_top_latest)
    state_top_values = create_value_lookup(state_top_df, state_top_latest)
    metro_bottom_values = create_value_lookup(metro_bottom_df, metro_bottom_latest)
    state_bottom_values = create_value_lookup(state_bottom_df, state_bottom_latest)

    # Apply Top Tier
    def get_top_tier(row):
        region_name = row['regionname'] # Define region_name here
        if row['regiontype'] == 'metro':
            return metro_top_values.get(region_name, None)
        elif row['regiontype'] == 'state':
            return state_top_values.get(region_name, None)
        # MODIFICATION START: Handle country
        elif row['regiontype'] == 'country':
            value = state_top_values.get(region_name, None) # Prefer state
            if value is None:
                value = metro_top_values.get(region_name, None) # Fallback metro
            return value
        # MODIFICATION END
        return None

    output_df['top_tier'] = output_df.apply(get_top_tier, axis=1)

    # Apply Bottom Tier
    def get_bottom_tier(row):
        region_name = row['regionname'] # Define region_name here
        if row['regiontype'] == 'metro':
            return metro_bottom_values.get(region_name, None)
        elif row['regiontype'] == 'state':
            return state_bottom_values.get(region_name, None)
        # MODIFICATION START: Handle country
        elif row['regiontype'] == 'country':
            value = state_bottom_values.get(region_name, None) # Prefer state
            if value is None:
                value = metro_bottom_values.get(region_name, None) # Fallback metro
            return value
        # MODIFICATION END
        return None

    output_df['bottom_tier'] = output_df.apply(get_bottom_tier, axis=1)

    logger.info("Finished processing price tiers.")
    return output_df

def process_sale_listing_prices(output_df, logger):
    """Process sale and listing price data and add to output DataFrame."""
    logger.info("Processing sale and listing prices...")

    try:
        # Load sale price data (metro only specified)
        sale_price_df = pd.read_csv('Metro_median_sale_price_uc_sfr_month.csv')
        # Load listing price data (metro only specified)
        listing_price_df = pd.read_csv('Metro_mlp_uc_sfrcondo_sm_month.csv')
    except FileNotFoundError as e:
        logger.error(f"Error loading sale/listing price file: {e}.")
        output_df['sale_price'] = np.nan
        output_df['listing_price'] = np.nan
        return output_df

    _, sale_latest = get_latest_date_columns(sale_price_df)
    _, listing_latest = get_latest_date_columns(listing_price_df)

    # Create lookups (includes US if present in metro files)
    sale_values = create_value_lookup(sale_price_df, sale_latest)
    listing_values = create_value_lookup(listing_price_df, listing_latest)

    # Apply Sale Price
    def get_sale_price(row):
        region_name = row['regionname'] # Define region_name
        # MODIFICATION START: Check country also
        if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
            return sale_values.get(region_name, None)
        # MODIFICATION END
        return None

    output_df['sale_price'] = output_df.apply(get_sale_price, axis=1)

    # Apply Listing Price
    def get_listing_price(row):
        region_name = row['regionname'] # Define region_name
        # MODIFICATION START: Check country also
        if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
            return listing_values.get(region_name, None)
        # MODIFICATION END
        return None

    output_df['listing_price'] = output_df.apply(get_listing_price, axis=1)

    logger.info("Finished processing sale and listing prices.")
    return output_df

def process_days_to_pending(output_df, logger):
    """Process days to pending data and add to output DataFrame."""
    logger.info("Processing days to pending data...")

    try:
        # Load days to pending data (Metro only specified)
        dtp_df = pd.read_csv('Metro_med_doz_pending_uc_sfrcondo_sm_month.csv')
    except FileNotFoundError as e:
        logger.error(f"Error loading DTP file: {e}.")
        output_df['current_days_to_pending'] = np.nan
        output_df['dtp_month'] = None
        output_df['days_to_pending'] = np.nan
        return output_df

    dtp_date_cols, dtp_latest = get_latest_date_columns(dtp_df)

    if not dtp_latest:
        logger.warning("No days to pending data columns found.")
        output_df['current_days_to_pending'] = np.nan
        output_df['dtp_month'] = None
        output_df['days_to_pending'] = np.nan
        return output_df

    # Create lookup for current days to pending (includes US if present in dtp_df)
    dtp_values = create_value_lookup(dtp_df, dtp_latest)

    # Format dtp_month in "Month YY" format
    dtp_month_fmt = None
    if dtp_latest:
        try:
            date_obj = pd.to_datetime(dtp_latest) # Use pandas for robust parsing
            dtp_month_fmt = date_obj.strftime('%B %y')
        except ValueError:
            logger.warning(f"Could not format DTP date '{dtp_latest}'")
            dtp_month_fmt = None # Set to None if formatting fails

    # Calculate average of last 12 months for each region (includes US if present)
    dtp_avg_values = {}
    if len(dtp_date_cols) >= 12:
        last_12_cols = dtp_date_cols[-12:]
        # Use pandas for efficient grouped calculation
        dtp_df_indexed = dtp_df.set_index('RegionName')
        dtp_avg_values = dtp_df_indexed[last_12_cols].mean(axis=1, skipna=True).to_dict()
    else:
        logger.warning("Less than 12 months data for DTP average.")

    # Apply Current DTP
    def get_current_dtp(row):
        region_name = row['regionname'] # Define region_name
        # MODIFICATION START: Check country also
        if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
            return dtp_values.get(region_name, None)
        # MODIFICATION END
        return None

    output_df['current_days_to_pending'] = output_df.apply(get_current_dtp, axis=1)

    # Add dtp_month column (same for all rows)
    output_df['dtp_month'] = dtp_month_fmt

    # Apply Average DTP
    def get_avg_dtp(row):
        region_name = row['regionname'] # Define region_name
        # MODIFICATION START: Check country also
        if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
            return dtp_avg_values.get(region_name, None)
        # MODIFICATION END
        return None

    output_df['days_to_pending'] = output_df.apply(get_avg_dtp, axis=1)

    logger.info("Finished processing days to pending.")
    return output_df

def process_inventory(output_df, logger):
    """Process inventory data and add to output DataFrame."""
    logger.info("Processing inventory data...")

    try:
        # Load inventory data (Metro only specified)
        inventory_df = pd.read_csv('Metro_invt_fs_uc_sfrcondo_sm_month.csv')
    except FileNotFoundError as e:
        logger.error(f"Error loading inventory file: {e}.")
        output_df['inventory'] = np.nan
        return output_df

    _, inv_latest = get_latest_date_columns(inventory_df)

    if not inv_latest:
        logger.warning("No inventory data columns found.")
        output_df['inventory'] = np.nan
        return output_df

    # Create lookup for inventory values (includes US if present)
    inventory_values = create_value_lookup(inventory_df, inv_latest)

    # Apply to output DataFrame
    def get_inventory(row):
        region_name = row['regionname'] # Define region_name
        # MODIFICATION START: Check country also
        if row['regiontype'] == 'metro' or row['regiontype'] == 'country':
            return inventory_values.get(region_name, None)
        # MODIFICATION END
        return None

    output_df['inventory'] = output_df.apply(get_inventory, axis=1)

    logger.info("Finished processing inventory.")
    return output_df

def process_bedroom_values(output_df, logger):
    """Process home values by bedroom count and add to output DataFrame."""
    logger.info("Processing bedroom data...")

    # Define bedroom file pairs (metro and state)
    # Correct filenames based on user's original script structure
    bedroom_files = [
         ('Metro_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'State_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'home_value_one_bed'),
         ('Metro_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'State_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'home_value_two_bed'),
         ('Metro_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'State_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'home_value_three_bed'),
         ('Metro_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'State_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'home_value_four_bed'),
         # Assume 5+ bed file names follow the same pattern
         # Check actual Zillow filenames if this is incorrect
         ('Metro_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'State_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
          'home_value_five_bed')
    ]

    # Process each bedroom type
    for metro_file, state_file, col_name in bedroom_files:
        logger.info(f"Processing {col_name}...")
        try:
            # Load data
            metro_df = pd.read_csv(metro_file)
            state_df = pd.read_csv(state_file)
        except FileNotFoundError as e:
            logger.error(f"  -> Error loading file for {col_name}: {e}. Skipping.")
            output_df[col_name] = np.nan # Ensure column exists
            continue # Skip to next bedroom count

        # Get latest dates
        _, metro_latest = get_latest_date_columns(metro_df)
        _, state_latest = get_latest_date_columns(state_df)

        # Create lookup dictionaries (includes US if present)
        metro_values = create_value_lookup(metro_df, metro_latest)
        state_values = create_value_lookup(state_df, state_latest)

        # Apply to output DataFrame
        def get_bedroom_value(row):
            region_name = row['regionname'] # Define region_name
            if row['regiontype'] == 'metro':
                return metro_values.get(region_name, None)
            elif row['regiontype'] == 'state':
                return state_values.get(region_name, None)
            # MODIFICATION START: Handle country
            elif row['regiontype'] == 'country':
                value = state_values.get(region_name, None) # Prefer state
                if value is None:
                    value = metro_values.get(region_name, None) # Fallback metro
                return value
            # MODIFICATION END
            return None

        output_df[col_name] = output_df.apply(get_bedroom_value, axis=1)

    logger.info("Finished processing bedroom data.")
    return output_df

def create_readme_file(logger):
    """Create a README.md file for the GitHub repository."""
    logger.info("Creating README.md file...")
    
    readme_content = """# Zillow Data Organization Project

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
"""
    
    try:
        with open("README.md", "w") as f:
            f.write(readme_content)
        logger.info("README.md created successfully.")
    except Exception as e:
        logger.error(f"Error creating README.md: {e}")

def create_requirements_file(logger):
    """Create a requirements.txt file for package dependencies."""
    logger.info("Creating requirements.txt file...")
    
    requirements = """pandas>=1.3.0
numpy>=1.20.0
requests>=2.25.0
beautifulsoup4>=4.9.3
selenium>=4.0.0
undetected-chromedriver>=3.0.0
selenium-stealth>=1.0.0
"""
    
    try:
        with open("requirements.txt", "w") as f:
            f.write(requirements)
        logger.info("requirements.txt created successfully.")
    except Exception as e:
        logger.error(f"Error creating requirements.txt: {e}")

# Execute script if run directly
if __name__ == "__main__":
    main()

