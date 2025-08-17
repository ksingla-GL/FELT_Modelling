# FELT Token Financial Model v2.0
## Fresh Earth Land Token - Treasury & NAV Projection Model

---

## Quick Start

1. **Modify input parameters** in the `inputs/` folder (see Input Files section)
2. **Run the model** by double-clicking `run.exe`
3. **View results** in the `outputs/` folder

---

## Dashboard Setup & Launch

### Prerequisites
Ensure Python 3.8 or higher is installed on your system. To check:
- **Windows**: Open Command Prompt and type `python --version`
- **Mac**: Open Terminal and type `python3 --version`

If Python is not installed, download from [python.org](https://www.python.org/downloads/)

### Installation Steps

#### Windows (Command Prompt)
```bash
# Navigate to the project folder
cd path\to\FELT_Token_Model

# Install required packages
pip install streamlit pandas numpy plotly

# Run the dashboard
streamlit run db.py
Mac/Linux (Terminal)
bash# Navigate to the project folder
cd path/to/FELT_Token_Model

# Install required packages
pip3 install streamlit pandas numpy plotly

# Run the dashboard
streamlit run db.py
Accessing the Dashboard

After running the command, you'll see output like:
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501

The dashboard will automatically open in your default browser
If it doesn't open automatically, manually navigate to: http://localhost:8501
```
### Dashboard Features

Executive Dashboard: High-level KPIs and token metrics
Farm Portfolio: Individual farm performance analysis
Financial Analysis: P&L, NAV bridge, and treasury management
Token Metrics: DCF valuation and price evolution
Scenario Analysis: Compare different scenarios and sensitivity
Model Inputs: View all input assumptions
Detailed Reports: Export data and create custom analyses

### Stopping the Dashboard

Press Ctrl+C in the terminal/command prompt to stop the server

### Troubleshooting

"streamlit: command not found": Ensure streamlit is installed: pip install streamlit
Port already in use: Another app is using port 8501. Stop it or use: streamlit run db.py --server.port 8502
Module not found errors: Install missing packages: pip install pandas numpy plotly
Can't find db.py: Ensure you're in the correct directory containing the db.py file

## Overview

This model projects the financial performance of the FELT token over 10 years, simulating:
- Acquisition of 100 regenerative agriculture farms (10 farms per year)
- Revenue generation from multiple Changes of Practice (CoPs)
- Token price appreciation through DCF valuation
- Treasury management and self-funding capabilities

### Key Features
- **DCF-based token valuation** with declining growth rates
- **Three farm types**: Low, Medium, and High performance CoPs
- **No-dilution issuance**: New tokens issued at prevailing NAV
- **Self-funding threshold**: Automatic detection when treasury can fund growth
- **Scenario analysis**: Conservative, Base, and Aggressive cases

---

## Input Files

All input files are CSV format located in the `inputs/` folder. Edit with Excel or any text editor.

### 1. `1_portfolio_config.csv`
Core investment and valuation parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| land_cost_per_farm | 10,000,000 | Base land acquisition cost |
| dev_cost_per_farm | 4,000,000 | Development/project costs |
| hectares_per_farm | 1,000 | Farm size |
| operator_salary_cap | 250,000 | Maximum operator payment per farm/year |
| corporate_tax_rate | 0.30 | Tax rate on treasury earnings |
| discount_rate | 0.08 | DCF discount rate (8%) |
| growth_rate_yr1_2 | 0.25 | Growth rate for years 1-2 (25%) |
| growth_rate_yr3_5 | 0.20 | Growth rate for years 3-5 (20%) |
| growth_rate_yr6_8 | 0.15 | Growth rate for years 6-8 (15%) |
| growth_rate_yr9_10 | 0.10 | Growth rate for years 9-10 (10%) |

### 2. `2_acquisition_mix.csv`
Defines how many farms of each type are acquired per year:

| Year | Low CoPs | Medium CoPs | High CoPs |
|------|----------|-------------|-----------|
| 1-10 | 4 | 3 | 3 |

**Total must equal 10 farms per year**

### 3. `3_cop_revenue_timelines.csv`
Revenue volumes (units per hectare) by farm type and age:

| cop_type | stream | operational_age | units_per_hectare |
|----------|--------|-----------------|-------------------|
| low/medium/high | forestry/soil/biodiversity/beef/water | 1-10 | varies |

- **Forestry & Soil**: Carbon credits (ACCUs)
- **Biodiversity**: Biodiversity credits
- **Beef**: Head count per hectare
- **Water**: Megalitres per hectare

### 4. `4_cost_structure.csv`
Stakeholder allocation percentages (must sum to 100%):

| Stakeholder | Default % | Description |
|-------------|-----------|-------------|
| expert | 5% | Agricultural consultants |
| supplier | 35% | Equipment and operations |
| operator | 10% | Farm operators (capped) |
| project_development | 15% | FEAG profit (PDF) |
| admin | 3% | Administration |
| treasury_pretax | 32% | Growth capital (22.4% after tax) |

### 5. `5_market_prices.csv`
Asset prices and growth rates:

| Asset | Spot Price | Annual Growth |
|-------|------------|---------------|
| forestry | 45 | 3% | 
| soil | 45 | 3% |
| biodiversity | 12,000 | 4% |
| beef | 4,000 | 3% |
| water | 5,000 | 4% |

**Note**: ACCU prices (forestry/soil) are fixed at $45 as per requirements

---

## Output Files

Generated in the `outputs/` folder after running the model:

### Core Outputs

1. **`output_portfolio_summary.csv`**
   - Year-by-year portfolio metrics
   - Token price (DCF and NAV-based)
   - Treasury and land values
   - Self-funding status

2. **`output_executive_kpi.txt`**
   - Summary of key performance indicators
   - Year 10 token price and multiple
   - Farm profitability metrics
   - Target achievement status

3. **`output_nav_reconciliation.csv`**
   - Detailed NAV bridge showing:
   - Opening NAV → Operating results → Land appreciation → Closing NAV

4. **`output_farm_ledger.csv`**
   - Detailed P&L for each farm
   - Revenue by stream
   - Stakeholder distributions
   - Cumulative profits

5. **`output_token_metrics.csv`**
   - Token price evolution
   - NAV growth
   - Token multiples

### Analysis Outputs

6. **`scenario_comparison.csv`**
   - Results for Conservative/Base/Aggressive scenarios
   - Key metrics comparison

7. **`sensitivity_analysis.csv`**
   - Parameter sensitivity testing
   - Impact on token price and farm profits

8. **`validation_report.txt`**
   - Model integrity checks
   - Target achievement verification

---

## Key Metrics & Targets

### Expected Results (Base Case)
- **Token Multiple**: ~20-30x by Year 10
- **Farm Profit**: $15-18M per farm (with 22.4% net margin)
- **Self-Funding**: Achieved by Year 5-6
- **Total Farms**: 100 by Year 10
- **FEAG PDF**: ~$200-250M cumulative

### Model Validation Checks
 Treasury properly reduced when farms purchased  
 NAV = Land Value + Treasury (no double counting)  
 Cost percentages sum to 100%  
 Token price = NAV/tokens with DCF premium  

---

## Understanding DCF Valuation

The model uses **Discounted Cash Flow (DCF)** valuation to capture future value:

1. **Current NAV**: Land value + Treasury cash
2. **Future Cash Flows**: Projected profits from all farms
3. **Growth Rates**: 
   - Years 1-2: 25% (high growth phase)
   - Years 3-5: 20% (expansion phase)
   - Years 6-8: 15% (maturation phase)
   - Years 9-10: 10% (stabilization)
   - Beyond Year 10: 3% perpetual growth
4. **Token Price** = (Current NAV + PV of Future Cash) / Tokens Outstanding

This gives investors credit for future value while maintaining realistic assumptions.

---

## Customization Guide

### To Model Different Scenarios:

1. **Optimistic Case**: 
   - Increase prices in `5_market_prices.csv`
   - Increase revenue volumes in `3_cop_revenue_timelines.csv`
   - Increase land appreciation in `1_portfolio_config.csv`

2. **Conservative Case**:
   - Reduce growth rates in `1_portfolio_config.csv`
   - Lower revenue assumptions
   - Increase discount rate

3. **Different Scale**:
   - Change `farms_per_year` in acquisition schedule
   - Adjust `hectares_per_farm`
   - Modify initial capital accordingly

### Important Constraints:
- **Cost structure** must sum to 100%
- **Acquisition mix** must total 10 farms/year
- **ACCU prices** should remain at $45 (Australian market rate)
- **Operator cap** is fixed at $250k/farm/year

---

## Troubleshooting

### Common Issues:

1. **"File not found" error**
   - Ensure all 5 input CSV files are present in `inputs/` folder
   - Check file names match exactly (including .csv extension)

2. **Model doesn't run**
   - Verify `run.exe` is in the same folder as `inputs/` and `outputs/`
   - Check Windows Defender hasn't blocked the executable
   - Try "Run as Administrator" if permissions issue

3. **Unexpected results**
   - Verify cost percentages sum to exactly 100%
   - Check acquisition mix totals 10 farms per year
   - Review revenue timelines for reasonable values

4. **Excel formatting issues**
   - Save CSVs in standard CSV format, not Excel workbook
   - Use decimal points (.) not commas for numbers
   - Don't include currency symbols or percentages in values

---

## Technical Notes

- **Model Horizon**: 10 years with annual calculations
- **Initial Tokens**: 140 million (matching $140M initial raise)
- **Base Currency**: USD
- **Revenue Recognition**: Annual basis
- **Tax Treatment**: 30% corporate tax on treasury earnings
- **Land Appreciation**: Compounds annually plus Green Prints premium

---

## Support

For questions or issues with the model, please contact the development team with:
1. Description of the issue
2. Screenshots of any error messages
3. Your modified input files (if applicable)

---

## Version History

**v3.0** (Current)
- Initial release with DCF valuation
- 100 farms over 10 years
- Three CoP types (Low/Medium/High)
- Self-funding capability
- Scenario and sensitivity analysis

---
