# Sales Data Processing Pipeline

A Python-based data processing pipeline for cleaning, transforming, and analyzing structured sales data.

## Tech Stack
- Python
- Pandas

## Project Objective
To simulate a basic ETL workflow by reading raw sales data, cleaning it, transforming it, and generating business insights.

## Features
- Data cleaning and validation
- Duplicate record removal
- Data type conversion (dates, numeric fields)
- Feature engineering (Total Revenue, Month extraction)
- Revenue and product-level aggregation
- Export of cleaned dataset and summary report

## Workflow

1. Load raw sales CSV file
2. Clean and validate data
3. Transform dataset and create new features
4. Generate key metrics
5. Export processed data and summary report

## Project Structure

sales-data-pipeline/
│
├── sales_pipeline.py      # Main ETL script
├── requirements.txt       # Dependencies
├── .gitignore             # Ignored files

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Place `sales.csv` in the project directory.

3. Run the script:
   python sales_pipeline.py

## Output

- cleaned_sales.csv
- summary_report.txt

This project demonstrates foundational data engineering concepts such as data cleaning, transformation, aggregation, and export using Python and Pandas.
