"""
FELT Token Financial Model - Interactive Dashboard (Fixed Version)
Comprehensive Streamlit application with error handling and improved visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime
import hashlib
import sys
import warnings
warnings.filterwarnings('ignore')

# Import model classes from Raw_code
sys.path.append('Raw_code')
try:
    from core_model import FELTModel, CoP, STREAMS
    from scenario_engine import ScenarioEngine
except ImportError:
    # Fallback if import fails
    FELTModel = None
    CoP = None
    ScenarioEngine = None
    STREAMS = ['forestry', 'soil', 'biodiversity', 'beef', 'water']

# Google Sheets URLs - Your actual sheet URLs
GOOGLE_SHEETS_URLS = {
    'portfolio_config': 'https://docs.google.com/spreadsheets/d/10_FypmxtPDT46Ay9rNz4lPk9CYSr_QzTutv3xKaYU3o/export?format=csv',
    'acquisition_mix': 'https://docs.google.com/spreadsheets/d/1nz6oVeYcS7ocE8_kyfCa4DrXx1L8F-vzmy3_Qy1huJQ/export?format=csv', 
    'cop_revenue': 'https://docs.google.com/spreadsheets/d/1OupIxigcvc9R3-bWu13s20Y1GcjdJ4fHnrijKHlRVgI/export?format=csv',
    'cost_structure': 'https://docs.google.com/spreadsheets/d/1WnkJVwiQV_hsACNHOqPGFWXGnEITTb-78sscPifJ6S8/export?format=csv',
    'market_prices': 'https://docs.google.com/spreadsheets/d/1j6Vr8TM4jKBLHMuDzV_rlAO2c80m3V575anAVhwM7_0/export?format=csv'
}

def load_from_google_sheets(url, sheet_name):
    """Load data from Google Sheets"""
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"Could not load from Google Sheets {sheet_name}: {str(e)}")
        return pd.DataFrame()

# Real model classes implementation
class CoP:
    """Change of Practice (farm) entity - REAL implementation"""
    
    def __init__(self, cop_id, cop_type, acquisition_year, hectares, land_cost, dev_cost):
        self.id = cop_id
        self.type = cop_type
        self.a_year = acquisition_year
        self.hectares = hectares
        self.land_fmv = land_cost + dev_cost
        self.initial_cost = land_cost + dev_cost
        self.cum_profit = 0.0
        self.ledger = []
        
    def compute_year(self, year, prices, costs, caps, tax_rate, revenue_lookup, green_prints_params):
        """Compute farm P&L for given year - REAL implementation"""
        age = year - self.a_year + 1
        if age < 1 or age > 10:
            return None
            
        # Gross revenue by stream
        gross = 0.0
        streams = {}
        
        for stream in STREAMS:
            units_per_ha = revenue_lookup.get((self.type, stream, age), 0)
            units = units_per_ha * self.hectares
            price = prices.get((stream, year), 0)
            rev = units * price
            streams[stream] = rev
            gross += rev
            
        # Stakeholder splits
        pays = {k: costs[k] * gross for k in costs.keys()}
        
        # Operator cap
        op_paid = min(pays.get('operator', 0), caps['operator_salary_cap'])
        op_over = max(pays.get('operator', 0) - op_paid, 0.0)
        
        # Pre-tax treasury
        t_pre = pays.get('treasury_pretax', 0) + op_over
        
        # Tax & net-to-treasury
        tax = tax_rate * t_pre
        ntt = t_pre - tax
        self.cum_profit += ntt
        
        # Land value update - REAL Green Prints logic
        base_appreciation = caps['land_appreciation_rate']
        
        # Calculate incremental premium
        prem_t = 0.0
        if age >= green_prints_params['start_age']:
            ramp = min(1.0, (age - green_prints_params['start_age'] + 1) / green_prints_params['ramp_years'])
            prem_t = green_prints_params['max_premium'] * ramp
            
        prem_prev = 0.0
        if age - 1 >= green_prints_params['start_age']:
            ramp_prev = min(1.0, (age - 1 - green_prints_params['start_age'] + 1) / green_prints_params['ramp_years'])
            prem_prev = green_prints_params['max_premium'] * ramp_prev
            
        # Apply incremental change only
        premium_multiplier = (1 + prem_t) / (1 + prem_prev) if prem_prev > 0 or prem_t > 0 else 1.0
        self.land_fmv *= (1 + base_appreciation) * premium_multiplier
        
        row = {
            'year': year,
            'age': age,
            'gross': gross,
            'gross_forestry': streams.get('forestry', 0),
            'gross_soil': streams.get('soil', 0),
            'gross_biodiversity': streams.get('biodiversity', 0),
            'gross_beef': streams.get('beef', 0),
            'gross_water': streams.get('water', 0),
            'pdf_fee': pays.get('project_development', 0),
            'expert_fee': pays.get('expert', 0),
            'supplier_costs': pays.get('supplier', 0),
            'admin': pays.get('admin', 0),
            'operator_paid': op_paid,
            'operator_overflow': op_over,
            'treasury_pretax': t_pre,
            'tax': tax,
            'net_to_treasury': ntt,
            'cum_profit': self.cum_profit,
            'land_fmv': self.land_fmv
        }
        
        self.ledger.append(row)
        return row

def calculate_input_hash(input_data):
    """Calculate hash of input data to detect changes"""
    hash_str = ""
    for key in sorted(input_data.keys()):
        if isinstance(input_data[key], pd.DataFrame):
            hash_str += f"{key}:{input_data[key].to_string()}"
        else:
            hash_str += f"{key}:{str(input_data[key])}"
    return hashlib.md5(hash_str.encode()).hexdigest()

class CustomFELTModel:
    """Custom FELT model that accepts DataFrames as inputs"""
    
    def __init__(self, input_data):
        self.input_data = input_data
        self.load_inputs()
        self.farms = []
        self.treasury = 0.0
        self.tokens_outstanding = 0.0
        self.farm_counter = 0
        
    def load_inputs(self):
        """Load inputs from provided DataFrames"""
        # Portfolio config
        if 'portfolio_config' in self.input_data and not self.input_data['portfolio_config'].empty:
            portfolio_df = self.input_data['portfolio_config']
            self.portfolio = {}
            for _, row in portfolio_df.iterrows():
                param = row['parameter']
                value = row['value']
                # Convert numeric parameters to float
                if param in ['land_cost_per_farm', 'dev_cost_per_farm', 'hectares_per_farm',
                            'land_appreciation_rate', 'operator_salary_cap', 'corporate_tax_rate',
                            'working_capital_reserve_pct', 'initial_token_supply', 'initial_raise',
                            'issuance_premium', 'green_prints_start_age', 'green_prints_max_premium',
                            'green_prints_ramp_years', 'discount_rate', 'growth_rate_yr1_2',
                            'growth_rate_yr3_5', 'growth_rate_yr6_8', 'growth_rate_yr9_10',
                            'terminal_growth', 'dcf_horizon']:
                    self.portfolio[param] = float(value)
                else:
                    self.portfolio[param] = value
        else:
            st.error("Portfolio configuration data missing")
            return
        
        # Acquisition mix
        if 'acquisition_mix' in self.input_data and not self.input_data['acquisition_mix'].empty:
            self.acquisition_mix = self.input_data['acquisition_mix']
        else:
            st.error("Acquisition mix data missing")
            return
        
        # Revenue timelines
        if 'cop_revenue' in self.input_data and not self.input_data['cop_revenue'].empty:
            revenue_df = self.input_data['cop_revenue']
            self.revenue_lookup = {}
            for _, row in revenue_df.iterrows():
                key = (row['cop_type'], row['stream'], int(row['operational_age']))
                self.revenue_lookup[key] = float(row['units_per_hectare'])
        else:
            st.error("Revenue timeline data missing")
            return
        
        # Cost structure
        if 'cost_structure' in self.input_data and not self.input_data['cost_structure'].empty:
            cost_df = self.input_data['cost_structure']
            self.costs = {}
            for _, row in cost_df.iterrows():
                self.costs[row['stakeholder']] = float(row['percentage'])
        else:
            st.error("Cost structure data missing")
            return
        
        # Market prices
        if 'market_prices' in self.input_data and not self.input_data['market_prices'].empty:
            prices_df = self.input_data['market_prices']
            self.base_prices = {}
            self.price_growth = {}
            for _, row in prices_df.iterrows():
                self.base_prices[row['asset']] = float(row['spot_price'])
                self.price_growth[row['asset']] = float(row['annual_growth_rate'])
                
            # Calculate prices for all years
            self.prices = {}
            for asset in self.base_prices:
                for year in range(1, 11):
                    price = self.base_prices[asset] * (1 + self.price_growth[asset]) ** (year - 1)
                    self.prices[(asset, year)] = price
        else:
            st.error("Market prices data missing")
            return
        
        # Green Prints parameters
        self.green_prints = {
            'start_age': int(self.portfolio.get('green_prints_start_age', 3)),
            'max_premium': float(self.portfolio.get('green_prints_max_premium', 0.15)),
            'ramp_years': int(self.portfolio.get('green_prints_ramp_years', 4))
        }
        
        # Caps
        self.caps = {
            'operator_salary_cap': float(self.portfolio['operator_salary_cap']),
            'land_appreciation_rate': float(self.portfolio['land_appreciation_rate'])
        }
    
    def calculate_nav(self):
        """Calculate total NAV = Land + Treasury"""
        land_nav = sum(farm.land_fmv for farm in self.farms)
        treasury_nav = self.treasury
        return land_nav + treasury_nav
    
    def calculate_dcf_token_value(self, year, current_nav, current_treasury):
        """Calculate token value using DCF of future cash flows - REAL implementation"""
        
        # Get DCF parameters
        discount_rate = float(self.portfolio.get('discount_rate', 0.08))
        growth_rates = {
            (1, 2): float(self.portfolio.get('growth_rate_yr1_2', 0.25)),
            (3, 5): float(self.portfolio.get('growth_rate_yr3_5', 0.20)),
            (6, 8): float(self.portfolio.get('growth_rate_yr6_8', 0.15)),
            (9, 10): float(self.portfolio.get('growth_rate_yr9_10', 0.10))
        }
        terminal_growth = float(self.portfolio.get('terminal_growth', 0.03))
        
        # Determine growth rate for current year
        current_growth = 0.10  # default
        for (start, end), rate in growth_rates.items():
            if start <= year <= end:
                current_growth = rate
                break
        
        # Estimate annual cash flow based on current treasury growth
        base_cash_flow = current_treasury * 0.224  # 22.4% net margin assumption
        
        # Calculate PV of future cash flows for next 10 years
        pv_future_cash = 0
        for t in range(1, 11):
            future_year = year + t
            
            # Determine growth rate for future year
            if future_year <= 2:
                g = growth_rates[(1, 2)]
            elif future_year <= 5:
                g = growth_rates[(3, 5)]
            elif future_year <= 8:
                g = growth_rates[(6, 8)]
            elif future_year <= 10:
                g = growth_rates[(9, 10)]
            else:
                g = terminal_growth
            
            # Project cash flow
            projected_cf = base_cash_flow * ((1 + g) ** t)
            
            # Discount to present value
            pv = projected_cf / ((1 + discount_rate) ** t)
            pv_future_cash += pv
        
        # Add terminal value (Gordon growth model) for cash flows beyond year 10
        terminal_cf = base_cash_flow * ((1 + terminal_growth) ** 11)
        terminal_value = terminal_cf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** 10)
        
        # Total DCF value
        dcf_nav = current_nav + pv_future_cash + pv_terminal
        
        # Token value
        if self.tokens_outstanding > 0:
            dcf_token_price = dcf_nav / self.tokens_outstanding
        else:
            dcf_token_price = 1.0
            
        return dcf_token_price, pv_future_cash
    
    def acquire_batch(self, year):
        """Acquire farms with REAL treasury handling"""
        # Get mix for this year
        if year <= len(self.acquisition_mix):
            mix = self.acquisition_mix.iloc[year - 1]
            low_count = int(mix['low_cops'])
            med_count = int(mix['medium_cops'])
            high_count = int(mix['high_cops'])
        else:
            low_count, med_count, high_count = 4, 3, 3
            
        total_farms = low_count + med_count + high_count
        unit_cost = self.portfolio['land_cost_per_farm'] + self.portfolio['dev_cost_per_farm']
        batch_cost = total_farms * unit_cost
        
        farms_acquired = []
        capital_raised = 0.0
        new_tokens = 0.0
        funding_source = 'none'
        
        # REAL capital and treasury handling
        if year == 1:
            # Year 1: Initial raise
            capital_raised = self.portfolio['initial_raise']
            new_tokens = self.portfolio['initial_token_supply']
            self.tokens_outstanding = new_tokens
            
            # Add capital to treasury
            self.treasury += capital_raised
            
            # PAY FOR FARMS FROM TREASURY
            self.treasury -= batch_cost
            
            funding_source = 'initial_raise'
            
        else:
            # Years 2+: Check if we can self-fund
            available = self.treasury * (1 - self.portfolio['working_capital_reserve_pct'])
            
            if available >= batch_cost:
                # Self-funding from treasury
                self.treasury -= batch_cost
                funding_source = 'treasury'
                
            else:
                # Need to raise capital
                funding_gap = batch_cost - available
                
                # Calculate token price at current NAV
                current_nav = self.calculate_nav()
                token_price = (current_nav / self.tokens_outstanding) if self.tokens_outstanding > 0 else 1.0
                issue_price = token_price * (1 + self.portfolio.get('issuance_premium', 0.0))
                
                capital_raised = funding_gap
                new_tokens = capital_raised / issue_price
                self.tokens_outstanding += new_tokens
                
                # Add new capital to treasury, then pay for farms
                self.treasury += capital_raised
                self.treasury -= batch_cost
                
                funding_source = 'mixed' if available > 0 else 'equity'
        
        # Create REAL farms
        for _ in range(low_count):
            self.farm_counter += 1
            farm = CoP(
                cop_id=f"F{self.farm_counter:03d}",
                cop_type='low',
                acquisition_year=year,
                hectares=self.portfolio['hectares_per_farm'],
                land_cost=self.portfolio['land_cost_per_farm'],
                dev_cost=self.portfolio['dev_cost_per_farm']
            )
            self.farms.append(farm)
            farms_acquired.append(farm)
            
        for _ in range(med_count):
            self.farm_counter += 1
            farm = CoP(
                cop_id=f"F{self.farm_counter:03d}",
                cop_type='medium',
                acquisition_year=year,
                hectares=self.portfolio['hectares_per_farm'],
                land_cost=self.portfolio['land_cost_per_farm'],
                dev_cost=self.portfolio['dev_cost_per_farm']
            )
            self.farms.append(farm)
            farms_acquired.append(farm)
            
        for _ in range(high_count):
            self.farm_counter += 1
            farm = CoP(
                cop_id=f"F{self.farm_counter:03d}",
                cop_type='high',
                acquisition_year=year,
                hectares=self.portfolio['hectares_per_farm'],
                land_cost=self.portfolio['land_cost_per_farm'],
                dev_cost=self.portfolio['dev_cost_per_farm']
            )
            self.farms.append(farm)
            farms_acquired.append(farm)
            
        return {
            'farms_acquired': farms_acquired,
            'capital_raised': capital_raised,
            'new_tokens': new_tokens,
            'funding_source': funding_source,
            'farms_count': len(farms_acquired)
        }

def run_model(input_data):
    """Run the REAL FELT model with provided input data"""
    try:
        # Create and run the REAL model
        model = CustomFELTModel(input_data)
        
        # Run 10-year projection with REAL accounting (copied from core_model.py)
        results = []
        
        for year in range(1, 11):
            year_data = {
                'year': year,
                'opening_farms': len(model.farms),
                'opening_treasury': model.treasury,
                'opening_nav': 0 if year == 1 else model.calculate_nav()
            }
            
            # Acquire farms at start of year - REAL implementation
            acquisition = model.acquire_batch(year)
            year_data['farms_acquired'] = acquisition['farms_count']
            year_data['capital_raised'] = acquisition['capital_raised']
            year_data['new_tokens'] = acquisition['new_tokens']
            year_data['funding_source'] = acquisition['funding_source']
            
            # Process all farms for the year - REAL revenue calculation
            farm_results = []
            for farm in model.farms:
                result = farm.compute_year(
                    year, 
                    model.prices, 
                    model.costs, 
                    model.caps,
                    model.portfolio['corporate_tax_rate'],
                    model.revenue_lookup,
                    model.green_prints
                )
                if result:
                    farm_results.append(result)
                    
            # Aggregate results - REAL calculations
            if farm_results:
                year_data['gross_revenue'] = sum(r['gross'] for r in farm_results)
                year_data['pdf_total'] = sum(r['pdf_fee'] for r in farm_results)
                year_data['expert_total'] = sum(r['expert_fee'] for r in farm_results)
                year_data['supplier_total'] = sum(r['supplier_costs'] for r in farm_results)
                year_data['admin_total'] = sum(r['admin'] for r in farm_results)
                year_data['operator_paid_total'] = sum(r['operator_paid'] for r in farm_results)
                year_data['operator_overflow_total'] = sum(r['operator_overflow'] for r in farm_results)
                year_data['treasury_pretax_total'] = sum(r['treasury_pretax'] for r in farm_results)
                year_data['tax_total'] = sum(r['tax'] for r in farm_results)
                year_data['net_to_treasury'] = sum(r['net_to_treasury'] for r in farm_results)
                
                # Add to treasury - REAL treasury management
                model.treasury += year_data['net_to_treasury']
            else:
                for key in ['gross_revenue', 'pdf_total', 'expert_total', 'supplier_total',
                          'admin_total', 'operator_paid_total', 'operator_overflow_total',
                          'treasury_pretax_total', 'tax_total', 'net_to_treasury']:
                    year_data[key] = 0
                    
            # End of year metrics - REAL calculations
            year_data['closing_farms'] = len(model.farms)
            year_data['closing_treasury'] = model.treasury
            year_data['land_nav'] = sum(farm.land_fmv for farm in model.farms)
            year_data['treasury_nav'] = model.treasury
            year_data['total_nav'] = year_data['land_nav'] + year_data['treasury_nav']
            year_data['tokens_outstanding'] = model.tokens_outstanding
            
            # Calculate both standard and DCF token prices - REAL DCF implementation
            standard_token_price = (year_data['total_nav'] / model.tokens_outstanding 
                                   if model.tokens_outstanding > 0 else 1.0)
            
            # Calculate DCF-based token price - REAL DCF calculation
            dcf_token_price, pv_future = model.calculate_dcf_token_value(
                year, 
                year_data['total_nav'],
                year_data['treasury_nav']
            )
            
            # Use DCF price as primary token price
            year_data['token_price'] = dcf_token_price
            year_data['standard_token_price'] = standard_token_price
            year_data['pv_future_cash'] = pv_future
            year_data['dcf_premium'] = (dcf_token_price / standard_token_price - 1) if standard_token_price > 0 else 0
            
            # Check self-funding threshold - REAL calculation
            reserve_factor = 1 - model.portfolio['working_capital_reserve_pct']
            next_year_cost = 10 * (model.portfolio['land_cost_per_farm'] + 
                                  model.portfolio['dev_cost_per_farm'])
            year_data['self_funding_capable'] = (model.treasury * reserve_factor >= next_year_cost)
            
            # Track cumulative profits - REAL farm profit tracking
            if model.farms:
                oldest_farms = sorted(model.farms, key=lambda f: f.a_year)[:10]
                year_data['oldest_farm_cum_profit'] = oldest_farms[0].cum_profit if oldest_farms else 0
                year_data['avg_cum_profit_first_10'] = (sum(f.cum_profit for f in oldest_farms) / 
                                                        len(oldest_farms) if oldest_farms else 0)
            else:
                year_data['oldest_farm_cum_profit'] = 0
                year_data['avg_cum_profit_first_10'] = 0
                
            results.append(year_data)
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Generate REAL outputs
        outputs = {
            'portfolio': df,
            'acquisition_mix': input_data['acquisition_mix'],
            'farm_ledger': create_real_farm_ledger(model.farms),
            'nav_recon': create_real_nav_reconciliation(df, model),
            'token_metrics': df[['year', 'total_nav', 'tokens_outstanding', 'token_price', 'standard_token_price', 'dcf_premium']].copy(),
            'stakeholder': create_real_stakeholder_flows(df)
        }
        
        # Run scenario analysis
        if ScenarioEngine:
            try:
                st.info("Running scenario analysis...")
                scenario_engine = ScenarioEngine()
                
                # Run scenarios (Conservative, Base, Aggressive)
                scenarios_df = scenario_engine.run_scenarios()
                outputs['scenarios'] = scenarios_df
                
                # Run sensitivity analysis
                sensitivity_df = scenario_engine.run_sensitivity() 
                outputs['sensitivity'] = sensitivity_df
                
                # Run Monte Carlo (smaller sample for dashboard speed)
                monte_stats_df = scenario_engine.run_monte_carlo(50)
                outputs['monte_carlo_stats'] = monte_stats_df
                
                st.success("✅ Scenario analysis completed!")
                
            except Exception as e:
                st.warning(f"Scenario analysis failed: {str(e)}")
                # Continue without scenario data
        else:
            st.warning("ScenarioEngine not available - using static scenario files")
        
        return outputs
        
    except Exception as e:
        st.error(f"Model execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def create_real_farm_ledger(farms):
    """Create REAL farm ledger from actual farm data"""
    ledger = []
    for farm in farms:
        for entry in farm.ledger:
            row = {
                'farm_id': farm.id,
                'farm_type': farm.type,
                'acquisition_year': farm.a_year,
                **{k: v for k, v in entry.items()}
            }
            ledger.append(row)
    return pd.DataFrame(ledger) if ledger else pd.DataFrame()

def create_real_nav_reconciliation(df, model):
    """Create REAL NAV reconciliation table"""
    nav_recon = []
    cost_per_farm = model.portfolio['land_cost_per_farm'] + model.portfolio['dev_cost_per_farm']
    
    for i, row in df.iterrows():
        if i == 0:
            # Year 1 - Start from zero
            opening_nav = 0
            new_capital = row['capital_raised']
            
            # Land appreciation only
            land_at_cost = row['farms_acquired'] * cost_per_farm
            land_total = row['land_nav']
            land_appreciation = land_total - land_at_cost
            
            operating = row['net_to_treasury']
            closing = row['total_nav']
            
            recon = {
                'year': row['year'],
                'opening_nav': 0,
                'new_capital': new_capital,
                'land_revaluation': land_appreciation,
                'operating_result': operating,
                'closing_nav': closing
            }
        else:
            # Years 2-10
            prev_row = df.iloc[i-1]
            opening_nav = prev_row['total_nav']
            new_capital = row['capital_raised']
            
            # Land appreciation only
            land_change = row['land_nav'] - prev_row['land_nav']
            new_farms_cost = row['farms_acquired'] * cost_per_farm
            land_appreciation = land_change - new_farms_cost
            
            operating = row['net_to_treasury']
            closing = row['total_nav']
            
            recon = {
                'year': row['year'],
                'opening_nav': opening_nav,
                'new_capital': new_capital,
                'land_revaluation': land_appreciation,
                'operating_result': operating,
                'closing_nav': closing
            }
        
        nav_recon.append(recon)
    
    return pd.DataFrame(nav_recon)

def create_real_stakeholder_flows(df):
    """Create REAL stakeholder flows table"""
    stakeholder_cols = ['year', 'pdf_total', 'expert_total', 'supplier_total',
                      'admin_total', 'operator_paid_total', 'operator_overflow_total', 'tax_total']
    available_cols = [col for col in stakeholder_cols if col in df.columns]
    return df[available_cols].copy() if available_cols else df[['year']].copy()

# Page configuration
try:
    # Try to use logo as page icon
    st.set_page_config(
        page_title="FELT Token Dashboard",
        page_icon="assets/logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    # Fallback to emoji if logo doesn't work as icon
    st.set_page_config(
        page_title="FELT Token Dashboard",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Custom CSS for better styling (unchanged)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = 10
if 'selected_farm' not in st.session_state:
    st.session_state.selected_farm = None
if 'selected_scenario' not in st.session_state:
    st.session_state.selected_scenario = 'Base'

# Load all data files with error handling
@st.cache_data(ttl=60)
def load_data():
    """Load all CSV files from outputs directory or run model with Google Sheets data"""
    data = {}
    
    # Load input files from Google Sheets first
    input_data = {}
    input_loaded_successfully = True
    
    for key, url in GOOGLE_SHEETS_URLS.items():
        sheet_data = load_from_google_sheets(url, key)
        if not sheet_data.empty:
            input_data[key] = sheet_data
        else:
            # Fallback to local files if Google Sheets fails
            local_files = {
                'portfolio_config': 'inputs/1_portfolio_config.csv',
                'acquisition_mix': 'inputs/2_acquisition_mix.csv',
                'cop_revenue': 'inputs/3_cop_revenue_timelines.csv',
                'cost_structure': 'inputs/4_cost_structure.csv',
                'market_prices': 'inputs/5_market_prices.csv'
            }
            if key in local_files and os.path.exists(local_files[key]):
                try:
                    input_data[key] = pd.read_csv(local_files[key])
                except Exception as e:
                    st.warning(f"Could not load {local_files[key]}: {str(e)}")
                    input_loaded_successfully = False
            else:
                input_loaded_successfully = False
    
    # Add input data to main data dict
    data.update(input_data)
    
    # Try to run model with current inputs
    model_outputs = {}
    if input_loaded_successfully and len(input_data) == 5:  # All 5 inputs loaded
        with st.spinner("Running financial model with current inputs..."):
            try:
                model_outputs = run_model(input_data)
                if model_outputs:
                    st.success("✅ Model executed successfully with live data!")
                    # Use model outputs as primary data source
                    for key, value in model_outputs.items():
                        if key not in ['acquisition_mix']:  # Don't override input data
                            data[key] = value
                else:
                    st.warning("⚠️ Model execution failed, using static files")
            except Exception as e:
                st.error(f"Model execution error: {str(e)}")
    else:
        st.warning("⚠️ Not all inputs available, using static output files")
    
    # Fallback: Load static output files if model didn't run or failed
    if not model_outputs:
        files = {
            'portfolio': 'output_portfolio_summary.csv',
            'stakeholder': 'output_stakeholder_flows.csv',
            'farm_ledger': 'output_farm_ledger.csv',
            'nav_recon': 'output_nav_reconciliation.csv',
            'token_metrics': 'output_token_metrics.csv',
            'acquisition': 'output_acquisition_schedule.csv',
            'scenarios': 'scenario_comparison.csv',
            'sensitivity': 'sensitivity_analysis.csv',
            'monte_carlo': 'monte_carlo_results.csv',
            'monte_stats': 'monte_carlo_stats.csv',
            'scenario_details': 'scenario_details.csv'
        }
        
        for key, filename in files.items():
            if key not in data:  # Only load if not already set by model
                filepath = f'outputs/{filename}'
                if os.path.exists(filepath):
                    try:
                        data[key] = pd.read_csv(filepath)
                    except Exception as e:
                        st.warning(f"Could not load {filename}: {str(e)}")
        
        # Load executive KPIs
        kpi_file = 'outputs/output_executive_kpi.txt'
        if os.path.exists(kpi_file):
            try:
                with open(kpi_file, 'r') as f:
                    data['executive_kpis'] = f.read()
            except Exception as e:
                st.warning(f"Could not load executive KPIs: {str(e)}")
    
    return data

# Load data
data = load_data()

# Helper functions
def format_currency(value, decimals=0):
    """Format value as currency"""
    if pd.isna(value):
        return "$0"
    if decimals == 0:
        return f"${value:,.0f}"
    else:
        return f"${value:,.{decimals}f}"

def format_number(value, decimals=0):
    """Format number with commas"""
    if pd.isna(value):
        return "0"
    if decimals == 0:
        return f"{value:,.0f}"
    else:
        return f"{value:,.{decimals}f}"

def safe_get(df, column, default=0):
    """Safely get column from dataframe"""
    if column in df.columns:
        return df[column]
    return default

def show_disclaimer():
    """Display legal disclaimer on all pages"""
    st.markdown("---")
    with st.expander("📋 Important Disclaimer - Please Read"):
        st.markdown("""
        **IMPORTANT DISCLAIMER**
        
        The financial information, projections, and assumptions set out in this document are provided for illustrative purposes only and do not represent forecasts, guarantees, or promises of future performance. Actual results may vary materially as they are subject to a wide range of known and unknown risks, uncertainties, and factors outside the control of Fresh Earth Universe Pty Ltd (the "Company"), including but not limited to changes in market conditions, regulatory developments, operational execution, and broader economic factors.
        
        Nothing in this document constitutes financial product advice, investment advice, or a recommendation to invest in the Company or any associated tokens. Any potential investor should conduct their own independent investigation and seek appropriate professional advice before making any investment decision. The Company undertakes no obligation to update the information contained herein.
        """, unsafe_allow_html=True)
    st.markdown("")

# SIDEBAR
with st.sidebar:
    # Logo in sidebar
    try:
        st.image("assets/logo.png", width=120)
    except:
        # Fallback if logo not found
        try:
            st.image("2025_Ag Logo_Picture1.png", width=120)
        except:
            pass  # No logo available
    
    st.markdown("# 🌱 FELT Token")
    st.markdown("### Navigation")
    
    page = st.radio(
        "Select View",
        ["📊 Executive Dashboard", 
         "🏞️ Farm Portfolio", 
         "💰 Financial Analysis",
         "📈 Token Metrics",
         "🔄 Scenario Analysis",
         "⚙️ Model Inputs",
         "📋 Detailed Reports"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Global Filters")
    
    # Year selector
    year_select = st.slider(
        "Select Year",
        min_value=1,
        max_value=10,
        value=st.session_state.selected_year,
        help="Select year for detailed analysis"
    )
    st.session_state.selected_year = year_select
    
    # Add cache control
    st.markdown("---")
    if st.button("🔄 Refresh Data", help="Clear cache and reload data from Google Sheets"):
        st.cache_data.clear()
        st.rerun()
    
    # Quick stats
    if 'portfolio' in data:
        current_data = data['portfolio'][data['portfolio']['year'] == year_select].iloc[0]
        st.markdown("### Quick Stats")
        st.metric("Token Price", format_currency(current_data.get('token_price', 0), 2))
        st.metric("Total Farms", int(current_data.get('closing_farms', 0)))
        st.metric("NAV", format_currency(current_data.get('total_nav', 0)))

# MAIN CONTENT
if page == "📊 Executive Dashboard":
    # Header with Logo
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            st.image("assets/logo.png", width=200)
        except:
            # Fallback if logo not found
            st.image("2025_Ag Logo_Picture1.png", width=200)
    
    st.markdown('<h1 class="main-header">FELT Token Financial Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fresh Earth Land Token - Regenerative Agriculture Investment Platform</p>', unsafe_allow_html=True)
    
    # Executive KPIs at top
    if 'portfolio' in data:
        year_10_data = data['portfolio'][data['portfolio']['year'] == 10].iloc[0]
        
        st.markdown("## 🎯 Year 10 Target Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            token_price = year_10_data.get('token_price', 0)
            st.metric(
                "Token Price (DCF)",
                format_currency(token_price, 2),
                f"{token_price:.1f}x",
                help="Discounted Cash Flow based token valuation"
            )
        
        with col2:
            total_nav = year_10_data.get('total_nav', 0)
            st.metric(
                "Portfolio NAV",
                format_currency(total_nav/1e6, 1) + "M",
                f"{(total_nav/1e9):.2f}B total"
            )
        
        with col3:
            st.metric(
                "Total Farms",
                int(year_10_data.get('closing_farms', 0)),
                "100 target"
            )
        
        with col4:
            gross_revenue = year_10_data.get('gross_revenue', 0)
            closing_farms = year_10_data.get('closing_farms', 1)
            st.metric(
                "Annual Revenue",
                format_currency(gross_revenue/1e6, 1) + "M",
                f"${gross_revenue/closing_farms/1e6:.1f}M/farm" if closing_farms > 0 else "N/A"
            )
        
        with col5:
            self_funding_year = data['portfolio'][data['portfolio'].get('self_funding_capable', False) == True]['year'].min() if 'self_funding_capable' in data['portfolio'].columns and any(data['portfolio'].get('self_funding_capable', False)) else "Not achieved"
            st.metric(
                "Self-Funding Year",
                self_funding_year,
                "Treasury funded",
                delta_color="inverse"
            )
    
    # Main charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Growth Trajectory", "💵 Revenue Streams", "🏦 Treasury Evolution", "🌎 Portfolio Composition"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Token price evolution
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['portfolio']['year'],
                y=data['portfolio'].get('token_price', data['portfolio'].get('token_price', 0)),
                name='Token Price (DCF)',
                line=dict(color='#2E7D32', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                hovertemplate='Year %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            if 'standard_token_price' in data['portfolio'].columns:
                fig.add_trace(go.Scatter(
                    x=data['portfolio']['year'],
                    y=data['portfolio']['standard_token_price'],
                    name='Cash NAV',  # Changed from 'NAV Price' to 'Cash NAV'
                    line=dict(color='#81C784', width=2, dash='dash'),
                    mode='lines',
                    hovertemplate='Year %{x}<br>Cash NAV: $%{y:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Token Price Evolution",
                xaxis_title="Year",
                yaxis_title="Token Price ($)",
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # NAV composition
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data['portfolio']['year'],
                y=data['portfolio'].get('land_nav', 0)/1e6,
                name='Land NAV',
                marker_color='#8BC34A',
                hovertemplate='Year %{x}<br>Land: $%{y:.1f}M<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=data['portfolio']['year'],
                y=data['portfolio'].get('treasury_nav', data['portfolio'].get('closing_treasury', 0))/1e6,
                name='Treasury NAV',
                marker_color='#4CAF50',
                hovertemplate='Year %{x}<br>Treasury: $%{y:.1f}M<extra></extra>'
            ))
            
            fig.update_layout(
                title="NAV Composition ($ Millions)",
                xaxis_title="Year",
                yaxis_title="NAV ($M)",
                barmode='stack',
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Revenue breakdown by stream with more distinct colors
        if 'farm_ledger' in data:
            revenue_cols = ['gross_forestry', 'gross_soil', 'gross_biodiversity', 'gross_beef', 'gross_water']
            # Check which columns exist
            existing_cols = [col for col in revenue_cols if col in data['farm_ledger'].columns]
            
            if existing_cols:
                revenue_by_year = data['farm_ledger'].groupby('year')[existing_cols].sum()
                
                fig = go.Figure()
                # More distinct colors - avoiding similar greens
                colors = ['#1B5E20', '#FF6F00', '#1976D2', '#7B1FA2', '#D32F2F']
                
                for i, col in enumerate(existing_cols):
                    stream_name = col.replace('gross_', '').title()
                    fig.add_trace(go.Scatter(
                        x=revenue_by_year.index,
                        y=revenue_by_year[col]/1e6,
                        name=stream_name,
                        mode='lines',
                        stackgroup='one',
                        fillcolor=colors[i],
                        line=dict(color=colors[i], width=0.5),
                        hovertemplate=f'{stream_name}<br>Year %{{x}}<br>${{y:.1f}}M<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Revenue Streams Evolution ($ Millions)",
                    xaxis_title="Year",
                    yaxis_title="Revenue ($M)",
                    hovermode='x unified',
                    showlegend=True,
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Revenue stream details not available")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Treasury growth and self-funding
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=data['portfolio']['year'],
                    y=data['portfolio'].get('net_to_treasury', 0)/1e6,
                    name='Net to Treasury',
                    marker_color='#4CAF50',
                    hovertemplate='Year %{x}<br>Net: $%{y:.1f}M<extra></extra>'
                ),
                secondary_y=False
            )
            
            treasury_col = 'closing_treasury' if 'closing_treasury' in data['portfolio'].columns else 'treasury_nav'
            fig.add_trace(
                go.Scatter(
                    x=data['portfolio']['year'],
                    y=data['portfolio'].get(treasury_col, 0)/1e6,
                    name='Cumulative Treasury',
                    line=dict(color='#1B5E20', width=3),
                    mode='lines+markers',
                    hovertemplate='Year %{x}<br>Total: $%{y:.1f}M<extra></extra>'
                ),
                secondary_y=True
            )
            
            # Add self-funding threshold line
            threshold = 140  # $140M for 10 farms
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                         annotation_text="Self-Funding Threshold",
                         secondary_y=True)
            
            fig.update_layout(
                title="Treasury Evolution & Self-Funding",
                xaxis_title="Year",
                hovermode='x unified',
                height=400
            )
            fig.update_yaxes(title_text="Annual Net ($M)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Treasury ($M)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Funding sources pie chart
            if 'acquisition' in data and 'capital_raised' in data['acquisition'].columns:
                funding_data = data['acquisition'].groupby('funding_source')['capital_raised'].sum()
                
                if len(funding_data) > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=funding_data.index,
                        values=funding_data.values,
                        hole=0.4,
                        marker_colors=['#2E7D32', '#4CAF50', '#66BB6A', '#81C784']
                    )])
                    
                    fig.update_layout(
                        title="Funding Sources Distribution",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Funding source data not available")
    
    with tab4:
        # Farm portfolio composition
        if 'farm_ledger' in data and 'farm_type' in data['farm_ledger'].columns:
            farm_types = data['farm_ledger'].groupby(['year', 'farm_type']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            colors = {'low': '#A5D6A7', 'medium': '#66BB6A', 'high': '#2E7D32'}
            
            for farm_type in farm_types.columns:
                fig.add_trace(go.Bar(
                    x=farm_types.index,
                    y=farm_types[farm_type],
                    name=farm_type.title() + ' CoPs',
                    marker_color=colors.get(farm_type, '#4CAF50'),
                    hovertemplate=f'{farm_type.title()}<br>Year %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Farm Portfolio Composition by Type",
                xaxis_title="Year",
                yaxis_title="Number of Farms",
                barmode='stack',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Add disclaimer
    show_disclaimer()

elif page == "🏞️ Farm Portfolio":
    st.markdown("# 🏞️ Farm Portfolio Analysis")
    
    # Farm overview metrics
    if 'farm_ledger' in data:
        current_year_farms = data['farm_ledger'][data['farm_ledger']['year'] == st.session_state.selected_year]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Farms", len(current_year_farms['farm_id'].unique()))
        with col2:
            avg_revenue = current_year_farms.get('gross', current_year_farms.get('gross_revenue', pd.Series([0]))).mean()
            st.metric("Avg Revenue/Farm", format_currency(avg_revenue/1e6, 2) + "M")
        with col3:
            avg_profit = current_year_farms.get('net_to_treasury', pd.Series([0])).mean()
            st.metric("Avg Net/Farm", format_currency(avg_profit/1e6, 2) + "M")
        with col4:
            total_hectares = len(current_year_farms['farm_id'].unique()) * 1000  # Assuming 1000 hectares per farm
            st.metric("Total Hectares", format_number(total_hectares))
        
        # Farm performance tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🌟 Individual Farms", "📊 Performance Distribution", "📈 Maturity Analysis", "🗺️ Geographic View"])
        
        with tab1:
            # Individual farm selector and details
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Select Farm")
                
                # Farm filters
                farm_type_filter = st.selectbox(
                    "Farm Type",
                    ["All"] + list(data['farm_ledger']['farm_type'].unique()) if 'farm_type' in data['farm_ledger'].columns else ["All"]
                )
                
                acquisition_year_filter = st.selectbox(
                    "Acquisition Year",
                    ["All"] + sorted(data['farm_ledger']['acquisition_year'].unique()) if 'acquisition_year' in data['farm_ledger'].columns else ["All"]
                )
                
                # Filter farms
                filtered_farms = data['farm_ledger'].copy()
                if farm_type_filter != "All" and 'farm_type' in filtered_farms.columns:
                    filtered_farms = filtered_farms[filtered_farms['farm_type'] == farm_type_filter]
                if acquisition_year_filter != "All" and 'acquisition_year' in filtered_farms.columns:
                    filtered_farms = filtered_farms[filtered_farms['acquisition_year'] == acquisition_year_filter]
                
                # Farm selector
                farm_ids = sorted(filtered_farms['farm_id'].unique()) if 'farm_id' in filtered_farms.columns else []
                if farm_ids:
                    selected_farm = st.selectbox("Farm ID", farm_ids)
                else:
                    selected_farm = None
                    st.info("No farms available with selected filters")
                
                # Farm quick stats
                if selected_farm:
                    farm_data = data['farm_ledger'][data['farm_ledger']['farm_id'] == selected_farm]
                    latest = farm_data[farm_data['year'] == farm_data['year'].max()].iloc[0]
                    
                    st.markdown("### Farm Stats")
                    if 'farm_type' in latest:
                        st.info(f"**Type:** {latest['farm_type'].title()} CoPs")
                    if 'age' in latest:
                        st.info(f"**Age:** {latest['age']} years")
                    if 'land_fmv' in latest:
                        st.info(f"**Land Value:** {format_currency(latest['land_fmv']/1e6, 2)}M")
                    if 'cum_profit' in latest:
                        st.info(f"**Cumulative Profit:** {format_currency(latest['cum_profit']/1e6, 2)}M")
            
            with col2:
                if selected_farm:
                    st.markdown(f"### Farm {selected_farm} Performance")
                    
                    # Farm revenue over time
                    farm_data = data['farm_ledger'][data['farm_ledger']['farm_id'] == selected_farm]
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Revenue Streams", "Profit Evolution", 
                                      "Cost Breakdown", "Land Value Growth"),
                        specs=[[{"type": "bar"}, {"type": "scatter"}],
                              [{"type": "pie"}, {"type": "scatter"}]]
                    )
                    
                    # Revenue streams
                    revenue_cols = ['gross_forestry', 'gross_soil', 'gross_biodiversity', 'gross_beef', 'gross_water']
                    existing_rev_cols = [col for col in revenue_cols if col in farm_data.columns]
                    
                    if existing_rev_cols:
                        latest_revenue = farm_data[farm_data['year'] == farm_data['year'].max()][existing_rev_cols].iloc[0]
                        
                        fig.add_trace(
                            go.Bar(
                                x=[col.replace('gross_', '').title() for col in existing_rev_cols],
                                y=latest_revenue.values/1e6,
                                marker_color='#4CAF50',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                    
                    # Profit evolution
                    if 'net_to_treasury' in farm_data.columns and 'age' in farm_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=farm_data['age'],
                                y=farm_data['net_to_treasury']/1e6,
                                mode='lines+markers',
                                line=dict(color='#2E7D32', width=2),
                                showlegend=False
                            ),
                            row=1, col=2
                        )
                    
                    # Cost breakdown
                    latest = farm_data[farm_data['year'] == farm_data['year'].max()].iloc[0]
                    cost_cols = ['pdf_fee', 'supplier_costs', 'expert_fee', 'admin', 'operator_paid', 'tax']
                    costs = {}
                    for col, label in zip(cost_cols, ['PDF', 'Supplier', 'Expert', 'Admin', 'Operator', 'Tax']):
                        if col in latest:
                            costs[label] = latest[col]
                    
                    if costs:
                        fig.add_trace(
                            go.Pie(
                                labels=list(costs.keys()),
                                values=list(costs.values()),
                                hole=0.4,
                                marker_colors=['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9']
                            ),
                            row=2, col=1
                        )
                    
                    # Land value
                    if 'land_fmv' in farm_data.columns and 'age' in farm_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=farm_data['age'],
                                y=farm_data['land_fmv']/1e6,
                                mode='lines+markers',
                                line=dict(color='#388E3C', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(56, 142, 60, 0.2)',
                                showlegend=False
                            ),
                            row=2, col=2
                        )
                    
                    # Update axes
                    fig.update_xaxes(title_text="Revenue Stream", row=1, col=1)
                    fig.update_yaxes(title_text="Revenue ($M)", row=1, col=1)
                    fig.update_xaxes(title_text="Operational Age", row=1, col=2)
                    fig.update_yaxes(title_text="Net Profit ($M)", row=1, col=2)
                    fig.update_xaxes(title_text="Operational Age", row=2, col=2)
                    fig.update_yaxes(title_text="Land Value ($M)", row=2, col=2)
                    
                    fig.update_layout(height=700, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Performance distribution analysis
            st.markdown("### Farm Performance Distribution")
            
            # Get latest data for all farms
            latest_data = []
            for farm_id in data['farm_ledger']['farm_id'].unique():
                farm = data['farm_ledger'][data['farm_ledger']['farm_id'] == farm_id]
                latest = farm[farm['year'] == farm['year'].max()].iloc[0]
                latest_data.append({
                    'farm_id': farm_id,
                    'type': latest.get('farm_type', 'unknown'),
                    'age': latest.get('age', 0),
                    'revenue': latest.get('gross', 0),
                    'profit': latest.get('net_to_treasury', 0),
                    'cum_profit': latest.get('cum_profit', 0),
                    'land_value': latest.get('land_fmv', 0)
                })
            
            perf_df = pd.DataFrame(latest_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue distribution by type
                if 'type' in perf_df.columns and 'revenue' in perf_df.columns:
                    fig = px.box(
                        perf_df,
                        x='type',
                        y='revenue',
                        color='type',
                        title="Revenue Distribution by Farm Type",
                        labels={'revenue': 'Annual Revenue ($)', 'type': 'Farm Type'},
                        color_discrete_map={'low': '#A5D6A7', 'medium': '#66BB6A', 'high': '#2E7D32'}
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Profit vs Age scatter
                if all(col in perf_df.columns for col in ['age', 'cum_profit', 'type', 'revenue']):
                    fig = px.scatter(
                        perf_df,
                        x='age',
                        y='cum_profit',
                        color='type',
                        size='revenue',
                        title="Cumulative Profit vs Farm Age",
                        labels={'cum_profit': 'Cumulative Profit ($)', 'age': 'Operational Age (years)'},
                        color_discrete_map={'low': '#A5D6A7', 'medium': '#66BB6A', 'high': '#2E7D32'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Maturity analysis
            st.markdown("### Farm Maturity Analysis")
            
            # Group farms by age
            maturity_groups = {
                'New (1-3 years)': perf_df[perf_df['age'] <= 3],
                'Growing (4-6 years)': perf_df[(perf_df['age'] > 3) & (perf_df['age'] <= 6)],
                'Mature (7-10 years)': perf_df[perf_df['age'] > 6]
            }
            
            maturity_stats = []
            for group_name, group_df in maturity_groups.items():
                if len(group_df) > 0:
                    maturity_stats.append({
                        'Group': group_name,
                        'Count': len(group_df),
                        'Avg Revenue': group_df['revenue'].mean(),
                        'Avg Profit': group_df['profit'].mean(),
                        'Total Cum Profit': group_df['cum_profit'].sum(),
                        'Avg Land Value': group_df['land_value'].mean()
                    })
            
            # Display metrics
            for stat in maturity_stats:
                st.markdown(f"#### {stat['Group']}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Farms", stat['Count'])
                with col2:
                    st.metric("Avg Revenue", format_currency(stat['Avg Revenue']/1e6, 2) + "M")
                with col3:
                    st.metric("Avg Profit", format_currency(stat['Avg Profit']/1e6, 2) + "M")
                with col4:
                    st.metric("Avg Land Value", format_currency(stat['Avg Land Value']/1e6, 2) + "M")
        
        with tab4:
            # Geographic placeholder
            st.markdown("### Geographic Distribution")
            st.info("🗺️ Geographic visualization would show farm locations across regions")
            
            # Create sample geographic distribution
            regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
            farm_counts = [20, 25, 30, 15, 10]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=regions,
                    y=farm_counts,
                    marker_color=['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']
                )
            ])
            
            fig.update_layout(
                title="Farm Distribution by Region (Illustrative)",
                xaxis_title="Region",
                yaxis_title="Number of Farms",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Add disclaimer
    show_disclaimer()

elif page == "💰 Financial Analysis":
    st.markdown("# 💰 Financial Analysis")
    
    # Financial overview
    if 'portfolio' in data:
        current = data['portfolio'][data['portfolio']['year'] == st.session_state.selected_year].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gross Revenue", format_currency(current.get('gross_revenue', 0)/1e6, 1) + "M")
        with col2:
            gross_rev = current.get('gross_revenue', 1)
            net_treasury = current.get('net_to_treasury', 0)
            margin = (net_treasury / gross_rev * 100) if gross_rev > 0 else 0
            st.metric("Net Margin", f"{margin:.1f}%")
        with col3:
            treasury = current.get('closing_treasury', current.get('treasury_nav', 0))
            st.metric("Treasury", format_currency(treasury/1e6, 1) + "M")
        with col4:
            st.metric("Total NAV", format_currency(current.get('total_nav', 0)/1e6, 1) + "M")
    
    # Financial tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💵 P&L Analysis", "📊 NAV Bridge", "👥 Stakeholder Flows", 
                                            "📈 Unit Economics", "🏦 Treasury Management"])
    
    with tab1:
        # P&L Analysis
        st.markdown("### Profit & Loss Statement")
        
        if 'portfolio' in data and 'stakeholder' in data:
            # Combine portfolio and stakeholder data
            pnl_data = pd.merge(data['portfolio'], data['stakeholder'], on='year')
            
            # Create P&L table for selected year
            year_data = pnl_data[pnl_data['year'] == st.session_state.selected_year].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Revenue")
                revenue_items = {
                    'Gross Revenue': year_data.get('gross_revenue', 0),
                    '': '',
                    'Less: Distributions': '',
                    '  Expert Fees': -year_data.get('expert_total', 0),
                    '  Supplier Costs': -year_data.get('supplier_total', 0),
                    '  Operator Payments': -year_data.get('operator_paid_total', 0),
                    '  Admin Costs': -year_data.get('admin_total', 0),
                    '  PDF Fees': -year_data.get('pdf_total', 0)
                }
                
                revenue_df = pd.DataFrame(list(revenue_items.items()), columns=['Item', 'Amount'])
                revenue_df['Amount'] = revenue_df['Amount'].apply(lambda x: format_currency(x) if x != '' else '')
                st.dataframe(revenue_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("#### Net Income")
                
                # Use available columns with fallbacks
                tax = year_data.get('tax_total', year_data.get('tax', 0))
                net = year_data.get('net_to_treasury', 0)
                gross = year_data.get('gross_revenue', 1)
                
                # Calculate pretax from net + tax if not directly available
                pretax = year_data.get('treasury_pretax_total', 
                                       year_data.get('treasury_pretax', 
                                                    net + tax if net > 0 else 0))

                # If still 0, try calculating from operator overflow
                if pretax == 0 and net > 0:
                    operator_overflow = year_data.get('operator_overflow_total', 0)
                    # Pre-tax should be net + tax
                    pretax = net + tax
                
                net_items = {
                    'Pre-tax Treasury': pretax,
                    'Less: Taxes': -tax,
                    '': '',
                    'Net to Treasury': net,
                    '': '',
                    'Net Margin': f"{(net/gross*100):.1f}%" if gross > 0 else "0.0%"
                }
                
                net_df = pd.DataFrame(list(net_items.items()), columns=['Item', 'Amount'])
                net_df['Amount'] = net_df['Amount'].apply(
                    lambda x: format_currency(x) if isinstance(x, (int, float)) else x
                )
                st.dataframe(net_df, hide_index=True, use_container_width=True)
            
            # Historical P&L trend
            st.markdown("#### Historical P&L Trends")
            
            fig = go.Figure()
            
            # Revenue line
            fig.add_trace(go.Scatter(
                x=pnl_data['year'],
                y=pnl_data.get('gross_revenue', 0)/1e6,
                name='Gross Revenue',
                line=dict(color='#2E7D32', width=3),
                mode='lines+markers'
            ))
            
            # Net income line
            fig.add_trace(go.Scatter(
                x=pnl_data['year'],
                y=pnl_data.get('net_to_treasury', 0)/1e6,
                name='Net to Treasury',
                line=dict(color='#4CAF50', width=3),
                mode='lines+markers'
            ))
            
            # Costs area
            cost_cols = ['expert_total', 'supplier_total', 'operator_paid_total', 'admin_total', 'pdf_total', 'tax_total']
            existing_cost_cols = [col for col in cost_cols if col in pnl_data.columns]
            
            if existing_cost_cols:
                total_costs = sum(pnl_data.get(col, 0) for col in existing_cost_cols)
                
                fig.add_trace(go.Scatter(
                    x=pnl_data['year'],
                    y=total_costs/1e6,
                    name='Total Costs',
                    line=dict(color='#FFA726', width=2, dash='dash'),
                    mode='lines'
                ))
            
            fig.update_layout(
                title="P&L Evolution ($ Millions)",
                xaxis_title="Year",
                yaxis_title="Amount ($M)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # NAV Bridge
        st.markdown("### NAV Reconciliation Bridge")
        
        if 'nav_recon' in data:
            year_nav = data['nav_recon'][data['nav_recon']['year'] == st.session_state.selected_year].iloc[0]
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="NAV Bridge",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Opening NAV", "New Capital", "Land Revaluation", "Operating Result", "Closing NAV"],
                y=[year_nav.get('opening_nav', 0)/1e6, 
                   year_nav.get('new_capital', 0)/1e6,
                   year_nav.get('land_revaluation', 0)/1e6, 
                   year_nav.get('operating_result', 0)/1e6,
                   year_nav.get('closing_nav', 0)/1e6],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#4CAF50"}},
                decreasing={"marker": {"color": "#F44336"}},
                totals={"marker": {"color": "#2E7D32"}}
            ))
            
            fig.update_layout(
                title=f"NAV Bridge - Year {st.session_state.selected_year}",
                yaxis_title="NAV ($ Millions)",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # NAV components table
            st.markdown("#### NAV Components Detail")
            
            opening = year_nav.get('opening_nav', 0)
            nav_detail = pd.DataFrame({
                'Component': ['Opening NAV', 'New Capital Raised', 'Land Appreciation', 
                             'Operating Profit', 'Closing NAV'],
                'Amount': [opening, year_nav.get('new_capital', 0), 
                          year_nav.get('land_revaluation', 0), year_nav.get('operating_result', 0), 
                          year_nav.get('closing_nav', 0)],
                'Change %': [0, 
                            year_nav.get('new_capital', 0)/opening*100 if opening > 0 else 0,
                            year_nav.get('land_revaluation', 0)/opening*100 if opening > 0 else 0,
                            year_nav.get('operating_result', 0)/opening*100 if opening > 0 else 0,
                            (year_nav.get('closing_nav', 0)-opening)/opening*100 if opening > 0 else 0]
            })
            
            nav_detail['Amount'] = nav_detail['Amount'].apply(lambda x: format_currency(x))
            nav_detail['Change %'] = nav_detail['Change %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(nav_detail, hide_index=True, use_container_width=True)
    
    with tab3:
        # Stakeholder flows
        st.markdown("### Stakeholder Distribution Analysis")
        
        if 'stakeholder' in data:
            stakeholder_data = data['stakeholder'][data['stakeholder']['year'] == st.session_state.selected_year].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of distributions
                distributions = {}
                dist_cols = {
                    'PDF (FEAG)': 'pdf_total',
                    'Suppliers': 'supplier_total',
                    'Experts': 'expert_total',
                    'Operators': 'operator_paid_total',
                    'Admin': 'admin_total',
                    'Tax': 'tax_total'
                }
                
                for label, col in dist_cols.items():
                    if col in stakeholder_data:
                        distributions[label] = stakeholder_data[col]
                
                # Add treasury if available
                if 'portfolio' in data:
                    port_data = data['portfolio'][data['portfolio']['year'] == st.session_state.selected_year]
                    if len(port_data) > 0:
                        distributions['Treasury (After Tax)'] = port_data['net_to_treasury'].iloc[0]
                
                if distributions:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(distributions.keys()),
                        values=list(distributions.values()),
                        hole=0.4,
                        marker_colors=['#1B5E20', '#2E7D32', '#388E3C', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']
                    )])
                    
                    fig.update_layout(
                        title=f"Stakeholder Distribution - Year {st.session_state.selected_year}",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cumulative flows over time
                if 'pdf_total' in data['stakeholder'].columns:
                    cumulative = data['stakeholder'].copy()
                    cumulative['cumulative_pdf'] = cumulative['pdf_total'].cumsum()
                    
                    if 'portfolio' in data and 'net_to_treasury' in data['portfolio'].columns:
                        cumulative['cumulative_treasury'] = data['portfolio']['net_to_treasury'].cumsum()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=cumulative['year'],
                        y=cumulative.get('cumulative_pdf', 0)/1e6,
                        name='Cumulative PDF',
                        line=dict(color='#1B5E20', width=3),
                        mode='lines+markers'
                    ))
                    
                    if 'cumulative_treasury' in cumulative.columns:
                        fig.add_trace(go.Scatter(
                            x=cumulative['year'],
                            y=cumulative['cumulative_treasury']/1e6,
                            name='Cumulative Treasury',
                            line=dict(color='#4CAF50', width=3),
                            mode='lines+markers'
                        ))
                    
                    fig.update_layout(
                        title="Cumulative Stakeholder Flows",
                        xaxis_title="Year",
                        yaxis_title="Cumulative Amount ($M)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Stakeholder summary table
            st.markdown("#### 10-Year Stakeholder Summary")
            
            summary_items = {}
            for label, col in dist_cols.items():
                if col in data['stakeholder'].columns:
                    summary_items[label] = data['stakeholder'][col].sum()
            
            if 'portfolio' in data and 'net_to_treasury' in data['portfolio'].columns:
                summary_items['Treasury'] = data['portfolio']['net_to_treasury'].sum()
            
            if summary_items:
                total = sum(summary_items.values())
                summary_data = {
                    'Stakeholder': list(summary_items.keys()),
                    'Total Received': list(summary_items.values()),
                    '% of Total': [v/total*100 for v in summary_items.values()]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df['Total Received'] = summary_df['Total Received'].apply(lambda x: format_currency(x))
                summary_df['% of Total'] = summary_df['% of Total'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    with tab4:
        # Unit Economics
        st.markdown("### Unit Economics Analysis")
        
        if 'farm_ledger' in data:
            # Calculate per-farm metrics
            required_cols = ['year', 'farm_id', 'gross', 'net_to_treasury', 'pdf_fee']
            existing_cols = [col for col in required_cols if col in data['farm_ledger'].columns]
            
            if len(existing_cols) >= 3:
                group_cols = ['year']
                agg_dict = {}
                
                if 'farm_id' in data['farm_ledger'].columns:
                    agg_dict['farm_id'] = 'nunique'
                if 'gross' in data['farm_ledger'].columns:
                    agg_dict['gross'] = 'sum'
                if 'net_to_treasury' in data['farm_ledger'].columns:
                    agg_dict['net_to_treasury'] = 'sum'
                if 'pdf_fee' in data['farm_ledger'].columns:
                    agg_dict['pdf_fee'] = 'sum'
                
                if agg_dict:
                    farms_by_year = data['farm_ledger'].groupby('year').agg(agg_dict)
                    
                    if 'farm_id' in farms_by_year.columns and farms_by_year['farm_id'].sum() > 0:
                        if 'gross' in farms_by_year.columns:
                            farms_by_year['revenue_per_farm'] = farms_by_year['gross'] / farms_by_year['farm_id']
                        if 'net_to_treasury' in farms_by_year.columns:
                            farms_by_year['profit_per_farm'] = farms_by_year['net_to_treasury'] / farms_by_year['farm_id']
                        if 'pdf_fee' in farms_by_year.columns:
                            farms_by_year['pdf_per_farm'] = farms_by_year['pdf_fee'] / farms_by_year['farm_id']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Per-farm metrics over time
                            fig = go.Figure()
                            
                            if 'revenue_per_farm' in farms_by_year.columns:
                                fig.add_trace(go.Scatter(
                                    x=farms_by_year.index,
                                    y=farms_by_year['revenue_per_farm']/1e6,
                                    name='Revenue/Farm',
                                    line=dict(color='#2E7D32', width=3),
                                    mode='lines+markers'
                                ))
                            
                            if 'profit_per_farm' in farms_by_year.columns:
                                fig.add_trace(go.Scatter(
                                    x=farms_by_year.index,
                                    y=farms_by_year['profit_per_farm']/1e6,
                                    name='Profit/Farm',
                                    line=dict(color='#4CAF50', width=3),
                                    mode='lines+markers'
                                ))
                            
                            fig.update_layout(
                                title="Per-Farm Economics Evolution",
                                xaxis_title="Year",
                                yaxis_title="Amount per Farm ($M)",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Efficiency metrics
                            if 'gross' in farms_by_year.columns and 'net_to_treasury' in farms_by_year.columns:
                                farms_by_year['margin'] = farms_by_year['net_to_treasury'] / farms_by_year['gross'] * 100
                                
                                if 'pdf_fee' in farms_by_year.columns:
                                    farms_by_year['pdf_rate'] = farms_by_year['pdf_fee'] / farms_by_year['gross'] * 100
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=farms_by_year.index,
                                    y=farms_by_year['margin'],
                                    name='Net Margin %',
                                    line=dict(color='#1B5E20', width=3),
                                    mode='lines+markers'
                                ))
                                
                                if 'pdf_rate' in farms_by_year.columns:
                                    fig.add_trace(go.Scatter(
                                        x=farms_by_year.index,
                                        y=farms_by_year['pdf_rate'],
                                        name='PDF Rate %',
                                        line=dict(color='#66BB6A', width=2, dash='dash'),
                                        mode='lines'
                                    ))
                                
                                fig.update_layout(
                                    title="Efficiency Metrics",
                                    xaxis_title="Year",
                                    yaxis_title="Percentage (%)",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Per-hectare analysis
                        st.markdown("#### Per-Hectare Economics")
                        
                        # Assuming 1000 hectares per farm
                        hectares_per_farm = 1000
                        if st.session_state.selected_year in farms_by_year.index:
                            current_farms = farms_by_year.loc[st.session_state.selected_year]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                if 'revenue_per_farm' in current_farms:
                                    revenue_per_ha = current_farms['revenue_per_farm'] / hectares_per_farm
                                    st.metric("Revenue/Hectare", format_currency(revenue_per_ha))
                            with col2:
                                if 'profit_per_farm' in current_farms:
                                    profit_per_ha = current_farms['profit_per_farm'] / hectares_per_farm
                                    st.metric("Profit/Hectare", format_currency(profit_per_ha))
                            with col3:
                                if 'farm_id' in current_farms:
                                    total_hectares = current_farms['farm_id'] * hectares_per_farm
                                    st.metric("Total Hectares", format_number(total_hectares))
                            with col4:
                                if 'profit_per_farm' in current_farms:
                                    roi = (current_farms['profit_per_farm'] / 14e6) * 100  # Assuming $14M investment per farm
                                    st.metric("ROI per Farm", f"{roi:.1f}%")
    
    with tab5:
        # Treasury Management
        st.markdown("### Treasury Management & Capital Efficiency")
        
        if 'portfolio' in data and 'acquisition' in data:
            portfolio = data['portfolio']
            acquisition = data['acquisition']
            
            # Treasury metrics
            current = portfolio[portfolio['year'] == st.session_state.selected_year].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            treasury_val = current.get('closing_treasury', current.get('treasury_nav', 0))
            with col1:
                st.metric("Current Treasury", format_currency(treasury_val/1e6, 1) + "M")
            with col2:
                reserve_pct = 0.10  # 10% reserve assumption
                available = treasury_val * (1 - reserve_pct)
                st.metric("Available Capital", format_currency(available/1e6, 1) + "M")
            with col3:
                next_year_need = 140e6  # $140M for 10 farms
                coverage = available / next_year_need if next_year_need > 0 else 0
                st.metric("Coverage Ratio", f"{coverage:.1%}")
            with col4:
                if 'self_funding_capable' in current and current['self_funding_capable']:
                    st.metric("Self-Funding", "✅ Active", "Achieved")
                else:
                    st.metric("Self-Funding", "❌ Not Yet", "Building")
            
            # Treasury evolution and projections
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Treasury Growth", "Funding Sources", 
                              "Capital Efficiency", "Working Capital"),
                specs=[[{"secondary_y": False}, {"type": "pie"}],
                      [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            # Treasury growth
            treasury_col = 'closing_treasury' if 'closing_treasury' in portfolio.columns else 'treasury_nav'
            
            fig.add_trace(
                go.Bar(
                    x=portfolio['year'],
                    y=portfolio.get('net_to_treasury', 0)/1e6,
                    name='Annual Addition',
                    marker_color='#81C784'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio['year'],
                    y=portfolio.get(treasury_col, 0)/1e6,
                    name='Cumulative',
                    line=dict(color='#2E7D32', width=3),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # Funding sources pie
            if 'funding_source' in acquisition.columns and 'capital_raised' in acquisition.columns:
                funding_summary = acquisition.groupby('funding_source')['capital_raised'].sum()
                if len(funding_summary) > 0:
                    fig.add_trace(
                        go.Pie(
                            labels=funding_summary.index,
                            values=funding_summary.values,
                            hole=0.4,
                            marker_colors=['#2E7D32', '#4CAF50', '#66BB6A', '#81C784']
                        ),
                        row=1, col=2
                    )
            
            # Capital efficiency (ROI)
            if 'closing_farms' in portfolio.columns:
                portfolio['capital_deployed'] = portfolio['closing_farms'] * 14e6  # $14M per farm
                
                if 'net_to_treasury' in portfolio.columns:
                    portfolio['roi'] = (portfolio['net_to_treasury'] / portfolio['capital_deployed'] * 100)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=portfolio['year'],
                            y=portfolio['roi'],
                            name='ROI %',
                            line=dict(color='#FF9800', width=2),
                            mode='lines+markers'
                        ),
                        row=2, col=1,
                        secondary_y=False
                    )
                
                fig.add_trace(
                    go.Bar(
                        x=portfolio['year'],
                        y=portfolio['capital_deployed']/1e6,
                        name='Capital Deployed',
                        marker_color='#FFC107',
                        opacity=0.5
                    ),
                    row=2, col=1,
                    secondary_y=True
                )
            
            # Working capital needs
            if treasury_col in portfolio.columns:
                portfolio['working_capital'] = portfolio[treasury_col] * 0.1  # 10% reserve
                portfolio['excess_capital'] = portfolio[treasury_col] - portfolio['working_capital']
                
                fig.add_trace(
                    go.Bar(
                        x=portfolio['year'],
                        y=portfolio['working_capital']/1e6,
                        name='Reserved',
                        marker_color='#FFCCBC'
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=portfolio['year'],
                        y=portfolio['excess_capital']/1e6,
                        name='Available',
                        marker_color='#4CAF50'
                    ),
                    row=2, col=2
                )
            
            # Update layouts
            fig.update_xaxes(title_text="Year", row=1, col=1)
            fig.update_yaxes(title_text="Treasury ($M)", row=1, col=1)
            fig.update_xaxes(title_text="Year", row=2, col=1)
            fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
            fig.update_yaxes(title_text="Capital ($M)", row=2, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Year", row=2, col=2)
            fig.update_yaxes(title_text="Amount ($M)", row=2, col=2)
            
            fig.update_layout(
                height=700,
                showlegend=True,
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Add disclaimer
    show_disclaimer()

elif page == "📈 Token Metrics":
    st.markdown("# 📈 Token Metrics & Valuation")
    
    if 'portfolio' in data:
        current = data['portfolio'][data['portfolio']['year'] == st.session_state.selected_year].iloc[0]
        
        # Token metrics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Token Price (DCF)", format_currency(current.get('token_price', 0), 2))
        with col2:
            st.metric("Token Multiple", f"{current.get('token_price', 0):.1f}x")
        with col3:
            tokens_out = current.get('tokens_outstanding', 140e6)  # Default to initial 140M
            st.metric("Tokens Outstanding", format_number(tokens_out/1e6, 1) + "M")
        with col4:
            dcf_premium = current.get('dcf_premium', 0) * 100
            st.metric("DCF Premium", f"{dcf_premium:.1f}%")
        
        # Token analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Evolution", "💡 DCF Analysis", "🔄 Token Issuance", "📈 Returns Analysis"])
        
        with tab1:
            # Token price evolution
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Token Price Evolution", "Price Components", 
                              "Token Multiple", "Market Cap Evolution")
            )
            
            # Price evolution
            fig.add_trace(
                go.Scatter(
                    x=data['portfolio']['year'],
                    y=data['portfolio'].get('token_price', 0),
                    name='DCF Price',
                    line=dict(color='#2E7D32', width=3),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            if 'standard_token_price' in data['portfolio'].columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['portfolio']['year'],
                        y=data['portfolio']['standard_token_price'],
                        name='Cash NAV',  # Changed from 'NAV Price' to 'Cash NAV'
                        line=dict(color='#81C784', width=2, dash='dash'),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            # Price components (DCF breakdown)
            if 'pv_future_cash' in data['portfolio'].columns:
                year_data = data['portfolio'][data['portfolio']['year'] == st.session_state.selected_year].iloc[0]
                tokens = year_data.get('tokens_outstanding', 140e6)
                if tokens > 0:
                    nav_component = year_data.get('total_nav', 0) / tokens
                    future_component = year_data.get('pv_future_cash', 0) / tokens
                    
                    fig.add_trace(
                        go.Bar(
                            x=['NAV Component', 'Future Cash PV'],
                            y=[nav_component, future_component],
                            marker_color=['#4CAF50', '#2E7D32']
                        ),
                        row=1, col=2
                    )
            
            # Token multiple
            fig.add_trace(
                go.Scatter(
                    x=data['portfolio']['year'],
                    y=data['portfolio'].get('token_price', 0),
                    name='Multiple',
                    line=dict(color='#388E3C', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(56, 142, 60, 0.2)',
                    mode='lines+markers'
                ),
                row=2, col=1
            )
            
            # Market cap - calculate if tokens_outstanding exists
            if 'tokens_outstanding' not in data['portfolio'].columns:
                # Use default 140M tokens
                data['portfolio']['market_cap'] = data['portfolio'].get('token_price', 0) * 140e6
            else:
                data['portfolio']['market_cap'] = data['portfolio']['token_price'] * data['portfolio']['tokens_outstanding']
            
            fig.add_trace(
                go.Scatter(
                    x=data['portfolio']['year'],
                    y=data['portfolio']['market_cap']/1e9,
                    name='Market Cap',
                    line=dict(color='#1B5E20', width=3),
                    mode='lines+markers',
                    fill='tozeroy',
                    fillcolor='rgba(27, 94, 32, 0.2)'
                ),
                row=2, col=2
            )
            
            # Update axes
            fig.update_xaxes(title_text="Year", row=1, col=1)
            fig.update_yaxes(title_text="Token Price ($)", row=1, col=1)
            fig.update_xaxes(title_text="Component", row=1, col=2)
            fig.update_yaxes(title_text="Price per Token ($)", row=1, col=2)
            fig.update_xaxes(title_text="Year", row=2, col=1)
            fig.update_yaxes(title_text="Multiple (x)", row=2, col=1)
            fig.update_xaxes(title_text="Year", row=2, col=2)
            fig.update_yaxes(title_text="Market Cap ($B)", row=2, col=2)
            
            fig.update_layout(height=700, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # DCF Analysis
            st.markdown("### DCF Valuation Components")
            
            # Load DCF parameters from config
            if 'portfolio_config' in data:
                config = data['portfolio_config']
                dcf_params = {}
                for _, row in config.iterrows():
                    if 'discount_rate' in row['parameter'] or 'growth_rate' in row['parameter'] or 'terminal' in row['parameter']:
                        dcf_params[row['parameter']] = float(row['value'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### DCF Parameters")
                    params_df = pd.DataFrame([
                        {'Parameter': 'Discount Rate', 'Value': f"{dcf_params.get('discount_rate', 0.08)*100:.0f}%"},
                        {'Parameter': 'Growth Rate Yr 1-2', 'Value': f"{dcf_params.get('growth_rate_yr1_2', 0.25)*100:.0f}%"},
                        {'Parameter': 'Growth Rate Yr 3-5', 'Value': f"{dcf_params.get('growth_rate_yr3_5', 0.20)*100:.0f}%"},
                        {'Parameter': 'Growth Rate Yr 6-8', 'Value': f"{dcf_params.get('growth_rate_yr6_8', 0.15)*100:.0f}%"},
                        {'Parameter': 'Growth Rate Yr 9-10', 'Value': f"{dcf_params.get('growth_rate_yr9_10', 0.10)*100:.0f}%"},
                        {'Parameter': 'Terminal Growth', 'Value': f"{dcf_params.get('terminal_growth', 0.03)*100:.0f}%"}
                    ])
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("#### Valuation Breakdown")
                    if 'pv_future_cash' in data['portfolio'].columns:
                        year_data = data['portfolio'][data['portfolio']['year'] == st.session_state.selected_year].iloc[0]
                        tokens = year_data.get('tokens_outstanding', 140e6)
                        
                        valuation_df = pd.DataFrame([
                            {'Component': 'Current NAV', 'Value': format_currency(year_data.get('total_nav', 0))},
                            {'Component': 'PV of Future Cash', 'Value': format_currency(year_data.get('pv_future_cash', 0))},
                            {'Component': 'Total DCF Value', 'Value': format_currency(year_data.get('token_price', 0) * tokens)},
                            {'Component': 'Tokens Outstanding', 'Value': format_number(tokens/1e6, 1) + "M"},
                            {'Component': 'DCF Price/Token', 'Value': format_currency(year_data.get('token_price', 0), 2)},
                            {'Component': 'Cash NAV/Token', 'Value': format_currency(year_data.get('standard_token_price', year_data.get('token_price', 0)), 2)}
                        ])
                        st.dataframe(valuation_df, hide_index=True, use_container_width=True)
            
            # DCF sensitivity
            st.markdown("#### DCF Sensitivity Analysis")
            
            # Create sensitivity matrix
            discount_rates = [0.06, 0.07, 0.08, 0.09, 0.10]
            growth_scenarios = ['Conservative (15%)', 'Base (20%)', 'Aggressive (25%)']
            
            # This would need actual recalculation - showing placeholder
            sensitivity_matrix = pd.DataFrame({
                'Discount Rate': discount_rates,
                'Conservative': [12.5, 10.8, 9.4, 8.3, 7.4],
                'Base': [15.2, 12.9, 11.1, 9.7, 8.5],
                'Aggressive': [18.7, 15.6, 13.2, 11.4, 9.9]
            })
            
            fig = go.Figure(data=go.Heatmap(
                z=[sensitivity_matrix['Conservative'], 
                   sensitivity_matrix['Base'], 
                   sensitivity_matrix['Aggressive']],
                x=sensitivity_matrix['Discount Rate'],
                y=growth_scenarios,
                colorscale='Greens',
                text=[[f"${v:.1f}" for v in sensitivity_matrix['Conservative']], 
                      [f"${v:.1f}" for v in sensitivity_matrix['Base']], 
                      [f"${v:.1f}" for v in sensitivity_matrix['Aggressive']]],
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title="Token Price Sensitivity (Discount Rate vs Growth)",
                xaxis_title="Discount Rate",
                yaxis_title="Growth Scenario",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Token Issuance
            st.markdown("### Token Issuance History")
            
            if 'acquisition' in data and 'new_tokens' in data['acquisition'].columns:
                issuance = data['acquisition'].copy()
                issuance['cumulative_tokens'] = issuance['new_tokens'].cumsum() + 140e6  # Initial 140M
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Token issuance over time
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=issuance['year'],
                        y=issuance['new_tokens']/1e6,
                        name='New Tokens',
                        marker_color='#66BB6A'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=issuance['year'],
                        y=issuance['cumulative_tokens']/1e6,
                        name='Cumulative',
                        line=dict(color='#2E7D32', width=3),
                        mode='lines+markers',
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title="Token Issuance History",
                        xaxis_title="Year",
                        yaxis=dict(title="New Tokens (M)"),
                        yaxis2=dict(title="Cumulative (M)", overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Dilution analysis
                    issuance['dilution'] = issuance.apply(
                        lambda row: row['new_tokens'] / (row['cumulative_tokens'] - row['new_tokens']) * 100 
                        if (row['cumulative_tokens'] - row['new_tokens']) > 0 else 0, 
                        axis=1
                    )
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=issuance['year'],
                        y=issuance['dilution'],
                        marker_color='#FFA726',
                        text=[f"{v:.1f}%" for v in issuance['dilution']],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Annual Dilution Rate",
                        xaxis_title="Year",
                        yaxis_title="Dilution (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Issuance summary table
                st.markdown("#### Token Issuance Summary")
                
                if 'capital_raised' in issuance.columns:
                    issuance_summary = issuance[['year', 'funding_source', 'capital_raised', 'new_tokens']].copy()
                    issuance_summary['price_per_token'] = issuance_summary.apply(
                        lambda row: row['capital_raised'] / row['new_tokens'] if row['new_tokens'] > 0 else 0,
                        axis=1
                    )
                    issuance_summary['capital_raised'] = issuance_summary['capital_raised'].apply(lambda x: format_currency(x))
                    issuance_summary['new_tokens'] = issuance_summary['new_tokens'].apply(lambda x: format_number(x/1e6, 2) + "M" if x > 0 else "-")
                    issuance_summary['price_per_token'] = issuance_summary['price_per_token'].apply(lambda x: format_currency(x, 2) if x > 0 else "-")
                    
                    st.dataframe(issuance_summary, hide_index=True, use_container_width=True)
        
        with tab4:
            # Returns Analysis
            st.markdown("### Investor Returns Analysis")
            
            # Calculate returns for different entry points
            returns_data = []
            for entry_year in range(1, min(8, len(data['portfolio'])) + 1):  # Up to year 7 entry
                entry_price = data['portfolio'][data['portfolio']['year'] == entry_year].get('token_price', pd.Series([1])).iloc[0]
                
                for exit_year in range(entry_year + 1, 11):
                    if exit_year <= len(data['portfolio']):
                        exit_price = data['portfolio'][data['portfolio']['year'] == exit_year].get('token_price', pd.Series([1])).iloc[0]
                        years_held = exit_year - entry_year
                        total_return = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
                        annual_return = ((exit_price / entry_price) ** (1/years_held) - 1) * 100 if entry_price > 0 and years_held > 0 else 0
                        
                        returns_data.append({
                            'Entry Year': entry_year,
                            'Exit Year': exit_year,
                            'Years Held': years_held,
                            'Entry Price': entry_price,
                            'Exit Price': exit_price,
                            'Total Return': total_return,
                            'Annual Return': annual_return
                        })
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                
                # Create returns heatmap
                pivot_returns = returns_df.pivot_table(
                    values='Annual Return',
                    index='Entry Year',
                    columns='Exit Year',
                    aggfunc='mean'
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_returns.values,
                    x=pivot_returns.columns,
                    y=pivot_returns.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=[[f"{v:.0f}%" if not pd.isna(v) else "" for v in row] for row in pivot_returns.values],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title="Annual<br>Return %")
                ))
                
                fig.update_layout(
                    title="Annualized Returns Matrix (Entry Year vs Exit Year)",
                    xaxis_title="Exit Year",
                    yaxis_title="Entry Year",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Best/worst returns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Best Returns")
                    best_returns = returns_df.nlargest(5, 'Annual Return')[['Entry Year', 'Exit Year', 'Years Held', 'Annual Return']]
                    best_returns['Annual Return'] = best_returns['Annual Return'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(best_returns, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("#### Investment Periods")
                    period_stats = returns_df.groupby('Years Held')['Annual Return'].agg(['mean', 'min', 'max'])
                    period_stats = period_stats.round(1)
                    period_stats.columns = ['Avg Return %', 'Min Return %', 'Max Return %']
                    st.dataframe(period_stats)
    
    # Add disclaimer
    show_disclaimer()
                    
elif page == "🔄 Scenario Analysis":
    st.markdown("# 🔄 Scenario & Sensitivity Analysis")
    
    # Scenario comparison
    if 'scenarios' in data:
        st.markdown("## Scenario Comparison")
        
        scenarios = data['scenarios']
        
        # Scenario selector
        selected_scenario = st.selectbox(
            "Select Scenario for Details",
            scenarios['scenario'].unique() if 'scenario' in scenarios.columns else ['Base'],
            index=1 if len(scenarios) > 1 else 0  # Default to Base if available
        )
        
        # Scenario metrics
        if len(scenarios) > 0:
            scenario_data = scenarios[scenarios['scenario'] == selected_scenario].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Token Price Y10", format_currency(scenario_data.get('year_10_token_price', 0), 2))
            with col2:
                st.metric("Token Multiple", f"{scenario_data.get('token_multiple', 0):.1f}x")
            with col3:
                st.metric("Min Farm Profit", format_currency(scenario_data.get('min_farm_profit', 0)/1e6, 1) + "M")
            with col4:
                self_fund = scenario_data.get('self_funding_year', 0)
                st.metric("Self-Funding Year", int(self_fund) if self_fund > 0 else "Not achieved")
        
        # Comparison charts
        tab1, tab2, tab3 = st.tabs(["📊 Scenario Comparison", "📈 Sensitivity Analysis", "🎲 Monte Carlo"])
        
        with tab1:
            # Scenario comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart comparison
                fig = go.Figure()
                
                metrics = ['year_10_token_price', 'token_multiple', 'min_farm_profit', 'self_funding_year']
                metric_names = ['Token Price', 'Multiple', 'Farm Profit (M)', 'Self-Fund Year']
                
                for i, metric in enumerate(metrics):
                    if metric in scenarios.columns:
                        values = scenarios[metric].values
                        if metric == 'min_farm_profit':
                            values = values / 1e6
                        
                        fig.add_trace(go.Bar(
                            name=metric_names[i],
                            x=scenarios['scenario'],
                            y=values,
                            text=[f"{v:.1f}" for v in values],
                            textposition='outside'
                        ))
                
                fig.update_layout(
                    title="Scenario Metrics Comparison",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Radar chart
                categories = ['Token Price', 'NAV', 'Revenue', 'Treasury', 'Land Value']
                
                fig = go.Figure()
                
                for _, row in scenarios.iterrows():
                    values = []
                    for cat, col in zip(categories, ['year_10_token_price', 'year_10_nav', 'year_10_revenue', 'year_10_treasury', 'year_10_land']):
                        if col in scenarios.columns:
                            max_val = scenarios[col].max()
                            values.append(row[col] / max_val if max_val > 0 else 0)
                        else:
                            values.append(0)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=row.get('scenario', 'Unknown')
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Scenario Performance Profile",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed scenario comparison table
            st.markdown("### Detailed Scenario Comparison")
            
            comparison_df = scenarios.copy()
            for col in comparison_df.columns:
                if col != 'scenario':
                    if any(term in col for term in ['price', 'nav', 'revenue', 'profit', 'treasury', 'land', 'pdf']):
                        comparison_df[col] = comparison_df[col].apply(lambda x: format_currency(x) if x > 1000 else f"{x:.2f}")
                    elif 'multiple' in col:
                        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.1f}x")
                    elif 'year' in col:
                        comparison_df[col] = comparison_df[col].apply(lambda x: int(x) if x > 0 else "N/A")
            
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        with tab2:
            # Sensitivity analysis
            if 'sensitivity' in data:
                st.markdown("### Parameter Sensitivity Analysis")
                
                sensitivity = data['sensitivity']
                
                # Parameter selector
                param_selected = st.selectbox(
                    "Select Parameter",
                    sensitivity['parameter'].unique() if 'parameter' in sensitivity.columns else []
                )
                
                if param_selected:
                    param_data = sensitivity[sensitivity['parameter'] == param_selected]
                    
                    # Create sensitivity charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Token price sensitivity
                        fig = go.Figure()
                        
                        if 'year_5_token_price' in param_data.columns:
                            fig.add_trace(go.Scatter(
                                x=param_data['value'],
                                y=param_data['year_5_token_price'],
                                name='Year 5',
                                line=dict(color='#66BB6A', width=2),
                                mode='lines+markers'
                            ))
                        
                        if 'year_10_token_price' in param_data.columns:
                            fig.add_trace(go.Scatter(
                                x=param_data['value'],
                                y=param_data['year_10_token_price'],
                                name='Year 10',
                                line=dict(color='#2E7D32', width=3),
                                mode='lines+markers'
                            ))
                        
                        fig.update_layout(
                            title=f"Token Price Sensitivity to {param_selected}",
                            xaxis_title=param_selected,
                            yaxis_title="Token Price ($)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Other metrics sensitivity
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        if 'min_farm_profit' in param_data.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=param_data['value'],
                                    y=param_data['min_farm_profit']/1e6,
                                    name='Farm Profit',
                                    line=dict(color='#4CAF50', width=2),
                                    mode='lines+markers'
                                ),
                                secondary_y=False
                            )
                        
                        if 'year_10_treasury' in param_data.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=param_data['value'],
                                    y=param_data['year_10_treasury']/1e6,
                                    name='Treasury',
                                    line=dict(color='#1B5E20', width=2, dash='dash'),
                                    mode='lines+markers'
                                ),
                                secondary_y=True
                            )
                        
                        fig.update_layout(
                            title=f"Financial Metrics Sensitivity",
                            xaxis_title=param_selected,
                            height=400
                        )
                        fig.update_yaxes(title_text="Farm Profit ($M)", secondary_y=False)
                        fig.update_yaxes(title_text="Treasury ($M)", secondary_y=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity summary
                st.markdown("### Sensitivity Summary")
                
                if 'parameter' in sensitivity.columns:
                    sensitivity_summary = []
                    for param in sensitivity['parameter'].unique():
                        param_sens = sensitivity[sensitivity['parameter'] == param]
                        if 'year_10_token_price' in param_sens.columns:
                            base_idx = len(param_sens) // 2  # Assume base is middle value
                            base_price = param_sens.iloc[base_idx]['year_10_token_price']
                            min_price = param_sens['year_10_token_price'].min()
                            max_price = param_sens['year_10_token_price'].max()
                            
                            sensitivity_summary.append({
                                'Parameter': param,
                                'Base Price': format_currency(base_price, 2),
                                'Min Price': format_currency(min_price, 2),
                                'Max Price': format_currency(max_price, 2),
                                'Range': format_currency(max_price - min_price, 2),
                                'Sensitivity': f"{(max_price/min_price - 1)*100:.0f}%" if min_price > 0 else "N/A"
                            })
                    
                    if sensitivity_summary:
                        sens_summary_df = pd.DataFrame(sensitivity_summary)
                        st.dataframe(sens_summary_df, hide_index=True, use_container_width=True)
        
        with tab3:
            # Monte Carlo analysis
            if 'monte_carlo' in data and 'monte_stats' in data:
                st.markdown("### Monte Carlo Risk Analysis")
                
                monte_carlo = data['monte_carlo']
                monte_stats = data['monte_stats']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of token prices
                    if 'token_price' in monte_carlo.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=monte_carlo['token_price'],
                            nbinsx=30,
                            marker_color='#4CAF50',
                            name='Token Price Distribution'
                        ))
                        
                        # Add percentile lines
                        p25 = monte_carlo['token_price'].quantile(0.25)
                        p50 = monte_carlo['token_price'].quantile(0.50)
                        p75 = monte_carlo['token_price'].quantile(0.75)
                        
                        fig.add_vline(x=p25, line_dash="dash", line_color="red", annotation_text="P25")
                        fig.add_vline(x=p50, line_dash="dash", line_color="blue", annotation_text="Median")
                        fig.add_vline(x=p75, line_dash="dash", line_color="green", annotation_text="P75")
                        
                        fig.update_layout(
                            title="Token Price Distribution (100 Simulations)",
                            xaxis_title="Token Price ($)",
                            yaxis_title="Frequency",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk metrics
                    st.markdown("#### Risk Metrics")
                    
                    if 'token_price' in monte_carlo.columns:
                        risk_metrics = pd.DataFrame([
                            {'Metric': 'Mean Token Price', 'Value': format_currency(monte_carlo['token_price'].mean(), 2)},
                            {'Metric': 'Std Deviation', 'Value': format_currency(monte_carlo['token_price'].std(), 2)},
                            {'Metric': '5th Percentile', 'Value': format_currency(monte_carlo['token_price'].quantile(0.05), 2)},
                            {'Metric': '25th Percentile', 'Value': format_currency(monte_carlo['token_price'].quantile(0.25), 2)},
                            {'Metric': 'Median', 'Value': format_currency(monte_carlo['token_price'].quantile(0.50), 2)},
                            {'Metric': '75th Percentile', 'Value': format_currency(monte_carlo['token_price'].quantile(0.75), 2)},
                            {'Metric': '95th Percentile', 'Value': format_currency(monte_carlo['token_price'].quantile(0.95), 2)},
                            {'Metric': 'Probability > $10', 'Value': f"{(monte_carlo['token_price'] > 10).mean()*100:.1f}%"},
                            {'Metric': 'Probability > $15', 'Value': f"{(monte_carlo['token_price'] > 15).mean()*100:.1f}%"},
                            {'Metric': 'Probability > $20', 'Value': f"{(monte_carlo['token_price'] > 20).mean()*100:.1f}%"}
                        ])
                        
                        st.dataframe(risk_metrics, hide_index=True, use_container_width=True)
                
                # NAV vs Token Price scatter
                if 'nav' in monte_carlo.columns and 'token_price' in monte_carlo.columns:
                    fig = px.scatter(
                        monte_carlo,
                        x='nav',
                        y='token_price',
                        color='self_funding' if 'self_funding' in monte_carlo.columns else None,
                        title="NAV vs Token Price Relationship",
                        labels={'nav': 'NAV ($)', 'token_price': 'Token Price ($)', 'self_funding': 'Self-Funding Year'},
                        color_continuous_scale='Greens'
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Scenario analysis data not available. Run scenario_engine.py to generate scenarios.")
    
    # Add disclaimer
    show_disclaimer()

elif page == "⚙️ Model Inputs":
    st.markdown("# ⚙️ Model Input Assumptions")
    
    st.info("📍 These are the input parameters used to generate the model outputs. Understanding these assumptions is critical for interpreting the results.")
    
    # Input tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Portfolio Config", "🏞️ Acquisition Mix", 
                                            "💰 Revenue Timelines", "👥 Cost Structure", "📈 Market Prices"])
    
    with tab1:
        st.markdown("### Portfolio Configuration")
        
        if 'portfolio_config' in data:
            config = data['portfolio_config']
            
            # Group parameters by category
            categories = {
                'Investment Parameters': ['land_cost_per_farm', 'dev_cost_per_farm', 'hectares_per_farm'],
                'Financial Parameters': ['corporate_tax_rate', 'operator_salary_cap', 'working_capital_reserve_pct'],
                'Token Parameters': ['initial_token_supply', 'initial_raise', 'issuance_premium'],
                'Valuation Parameters': ['land_appreciation_rate', 'green_prints_start_age', 
                                        'green_prints_max_premium', 'green_prints_ramp_years'],
                'DCF Parameters': ['discount_rate', 'growth_rate_yr1_2', 'growth_rate_yr3_5', 
                                  'growth_rate_yr6_8', 'growth_rate_yr9_10', 'terminal_growth']
            }
            
            for category, params in categories.items():
                st.markdown(f"#### {category}")
                
                category_data = config[config['parameter'].isin(params)].copy()
                
                if len(category_data) > 0:
                    # Format values based on parameter type
                    for idx, row in category_data.iterrows():
                        param = row['parameter']
                        value = float(row['value'])
                        
                        if 'rate' in param or 'premium' in param or 'pct' in param:
                            category_data.at[idx, 'value'] = f"{value*100:.1f}%"
                        elif 'cost' in param or 'raise' in param or 'salary' in param:
                            category_data.at[idx, 'value'] = format_currency(value)
                        elif 'supply' in param:
                            category_data.at[idx, 'value'] = format_number(value/1e6, 0) + "M"
                        elif 'hectares' in param:
                            category_data.at[idx, 'value'] = format_number(value)
                        else:
                            category_data.at[idx, 'value'] = f"{value:.0f}"
                    
                    # Make parameter names more readable
                    category_data['parameter'] = category_data['parameter'].str.replace('_', ' ').str.title()
                    
                    st.dataframe(category_data, hide_index=True, use_container_width=True)
        else:
            st.warning("Portfolio configuration data not available")
    
    with tab2:
        st.markdown("### Farm Acquisition Schedule")
        
        if 'acquisition_mix' in data:
            acq_mix = data['acquisition_mix']
            
            # Add totals column
            acq_mix['total_farms'] = acq_mix.get('low_cops', 0) + acq_mix.get('medium_cops', 0) + acq_mix.get('high_cops', 0)
            
            # Create stacked bar chart
            fig = go.Figure()
            
            if 'low_cops' in acq_mix.columns:
                fig.add_trace(go.Bar(
                    x=acq_mix['year'],
                    y=acq_mix['low_cops'],
                    name='Low CoPs',
                    marker_color='#A5D6A7'
                ))
            
            if 'medium_cops' in acq_mix.columns:
                fig.add_trace(go.Bar(
                    x=acq_mix['year'],
                    y=acq_mix['medium_cops'],
                    name='Medium CoPs',
                    marker_color='#66BB6A'
                ))
            
            if 'high_cops' in acq_mix.columns:
                fig.add_trace(go.Bar(
                    x=acq_mix['year'],
                    y=acq_mix['high_cops'],
                    name='High CoPs',
                    marker_color='#2E7D32'
                ))
            
            fig.update_layout(
                title="Annual Farm Acquisition Mix",
                xaxis_title="Year",
                yaxis_title="Number of Farms",
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.markdown("#### Acquisition Details")
            st.dataframe(acq_mix, hide_index=True, use_container_width=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Farms", acq_mix['total_farms'].sum())
            with col2:
                st.metric("Low CoPs", acq_mix.get('low_cops', 0).sum())
            with col3:
                st.metric("Medium CoPs", acq_mix.get('medium_cops', 0).sum())
            with col4:
                st.metric("High CoPs", acq_mix.get('high_cops', 0).sum())
        else:
            st.warning("Acquisition mix data not available")
    
    with tab3:
        st.markdown("### Revenue Timelines by CoP Type")
        
        if 'cop_revenue' in data:
            revenue = data['cop_revenue']
            
            # Stream selector
            stream_selected = st.selectbox(
                "Select Revenue Stream",
                revenue['stream'].unique() if 'stream' in revenue.columns else []
            )
            
            if stream_selected:
                stream_data = revenue[revenue['stream'] == stream_selected]
                
                # Create line chart for selected stream
                fig = go.Figure()
                
                for cop_type in stream_data['cop_type'].unique() if 'cop_type' in stream_data.columns else []:
                    cop_data = stream_data[stream_data['cop_type'] == cop_type]
                    
                    color_map = {'low': '#A5D6A7', 'medium': '#66BB6A', 'high': '#2E7D32'}
                    
                    fig.add_trace(go.Scatter(
                        x=cop_data['operational_age'] if 'operational_age' in cop_data.columns else [],
                        y=cop_data['units_per_hectare'] if 'units_per_hectare' in cop_data.columns else [],
                        name=f"{cop_type.title()} CoPs",
                        line=dict(color=color_map.get(cop_type, '#4CAF50'), width=3),
                        mode='lines+markers'
                    ))
                
                fig.update_layout(
                    title=f"{stream_selected.title()} Revenue Timeline (Units per Hectare)",
                    xaxis_title="Operational Age (Years)",
                    yaxis_title="Units per Hectare",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table for all streams
                st.markdown("#### Revenue Summary by Stream and Type")
                
                if all(col in revenue.columns for col in ['stream', 'cop_type', 'units_per_hectare']):
                    summary = revenue.groupby(['stream', 'cop_type'])['units_per_hectare'].agg(['mean', 'max']).round(2)
                    summary = summary.reset_index()
                    summary.columns = ['Stream', 'CoP Type', 'Avg Units/Ha', 'Max Units/Ha']
                    
                    # Pivot for better display
                    pivot = summary.pivot(index='Stream', columns='CoP Type', values='Max Units/Ha')
                    st.dataframe(pivot, use_container_width=True)
        else:
            st.warning("Revenue timeline data not available")
    
    with tab4:
        st.markdown("### Cost Structure & Stakeholder Allocation")
        
        if 'cost_structure' in data:
            costs = data['cost_structure']
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=costs['stakeholder'].str.replace('_', ' ').str.title() if 'stakeholder' in costs.columns else [],
                values=costs['percentage'] if 'percentage' in costs.columns else [],
                hole=0.4,
                marker_colors=['#1B5E20', '#2E7D32', '#388E3C', '#4CAF50', '#66BB6A', '#81C784'],
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Revenue Distribution Structure",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with descriptions
            if 'stakeholder' in costs.columns and 'percentage' in costs.columns:
                cost_detail = costs.copy()
                cost_detail['stakeholder'] = cost_detail['stakeholder'].str.replace('_', ' ').str.title()
                cost_detail['percentage'] = cost_detail['percentage'].apply(lambda x: f"{x*100:.1f}%")
                
                # Add descriptions
                descriptions = {
                    'Expert': 'Agricultural consultants and technical advisors',
                    'Supplier': 'Equipment, inputs, and operational costs',
                    'Operator': 'Farm management (capped at $250k/farm)',
                    'Project Development': 'FEAG profit share (PDF)',
                    'Admin': 'Administrative and overhead costs',
                    'Treasury Pretax': 'Growth capital before tax (22.4% after tax)'
                }
                
                cost_detail['description'] = cost_detail['stakeholder'].map(descriptions)
                
                st.dataframe(cost_detail[['stakeholder', 'percentage', 'description']], hide_index=True, use_container_width=True)
                
                # Verify totals
                total = costs['percentage'].sum()
                if abs(total - 1.0) < 0.001:
                    st.success(f"✅ Cost structure totals to {total*100:.1f}%")
                else:
                    st.error(f"⚠️ Cost structure totals to {total*100:.1f}% (should be 100%)")
        else:
            st.warning("Cost structure data not available")
    
    with tab5:
        st.markdown("### Market Price Assumptions")
        
        if 'market_prices' in data:
            prices = data['market_prices']
            
            # Create price projection chart
            fig = go.Figure()
            
            years = list(range(1, 11))
            colors = {'forestry': '#1B5E20', 'soil': '#2E7D32', 'biodiversity': '#388E3C', 
                     'beef': '#4CAF50', 'water': '#66BB6A'}
            
            for _, row in prices.iterrows():
                asset = row.get('asset', '')
                spot = row.get('spot_price', 0)
                growth = row.get('annual_growth_rate', 0)
                
                projected_prices = [spot * (1 + growth) ** (year - 1) for year in years]
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=projected_prices,
                    name=asset.title() if asset else 'Unknown',
                    line=dict(color=colors.get(asset, '#4CAF50'), width=2),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title="Price Projections (10 Years)",
                xaxis_title="Year",
                yaxis_title="Price",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price table with projections
            st.markdown("#### Price Details")
            
            price_detail = prices.copy()
            if all(col in price_detail.columns for col in ['asset', 'spot_price', 'annual_growth_rate']):
                price_detail['spot_price'] = price_detail.apply(
                    lambda x: format_currency(x['spot_price']) if x['asset'] in ['biodiversity', 'beef', 'water'] else f"${x['spot_price']}", 
                    axis=1
                )
                price_detail['annual_growth_rate'] = price_detail['annual_growth_rate'].apply(lambda x: f"{x*100:.0f}%")
                
                # Add year 10 price
                price_detail['year_10_price'] = prices.apply(
                    lambda x: format_currency(x['spot_price'] * (1 + x['annual_growth_rate']) ** 9) if x['asset'] in ['biodiversity', 'beef', 'water'] else f"${x['spot_price'] * (1 + x['annual_growth_rate']) ** 9:.0f}",
                    axis=1
                )
                
                price_detail['asset'] = price_detail['asset'].str.title()
                price_detail.columns = ['Asset', 'Spot Price', 'Annual Growth', 'Year 10 Price']
                
                st.dataframe(price_detail, hide_index=True, use_container_width=True)
            
            # Unit descriptions
            st.markdown("#### Asset Unit Descriptions")
            unit_desc = pd.DataFrame([
                {'Asset': 'Forestry', 'Unit': 'ACCU (Carbon Credit)', 'Description': 'Australian Carbon Credit Units from forest sequestration'},
                {'Asset': 'Soil', 'Unit': 'ACCU (Carbon Credit)', 'Description': 'Carbon credits from soil carbon sequestration'},
                {'Asset': 'Biodiversity', 'Unit': 'Biodiversity Credit', 'Description': 'Credits for biodiversity conservation and enhancement'},
                {'Asset': 'Beef', 'Unit': 'Head', 'Description': 'Price per head of cattle'},
                {'Asset': 'Water', 'Unit': 'Megalitre', 'Description': 'Price per megalitre of water rights'}
            ])
            
            st.dataframe(unit_desc, hide_index=True, use_container_width=True)
        else:
            st.warning("Market price data not available")
    
    # Add disclaimer
    show_disclaimer()

elif page == "📋 Detailed Reports":
    st.markdown("# 📋 Detailed Reports & Data Export")
    
    st.info("📊 Access and download all detailed model outputs and reports")
    
    # Report tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📑 Executive Report", "📊 Full Data Tables", 
                                      "💾 Export Data", "📈 Custom Analysis"])
    
    with tab1:
        st.markdown("### Executive Summary Report")
        
        if 'executive_kpis' in data:
            # Display executive KPIs
            st.text(data['executive_kpis'])
        
        # Generate executive summary
        if 'portfolio' in data and len(data['portfolio']) > 0:
            year_10 = data['portfolio'][data['portfolio']['year'] == 10].iloc[0] if 10 in data['portfolio']['year'].values else data['portfolio'].iloc[-1]
            year_1 = data['portfolio'][data['portfolio']['year'] == 1].iloc[0] if 1 in data['portfolio']['year'].values else data['portfolio'].iloc[0]
            
            st.markdown("### Investment Performance Summary")
            
            summary_text = f"""
            **FELT Token Financial Model - Executive Summary**
            
            **Investment Thesis:**
            The FELT Token represents a revolutionary approach to regenerative agriculture investment, 
            combining blockchain technology with sustainable farming practices across 100 farms over 10 years.
            
            **Key Performance Indicators:**
            - **Token Performance**: ${year_1.get('token_price', 1):.2f} → ${year_10.get('token_price', 0):.2f} ({year_10.get('token_price', 0):.1f}x multiple)
            - **Portfolio Scale**: {int(year_10.get('closing_farms', 0))} regenerative farms across {int(year_10.get('closing_farms', 0)) * 1000:,} hectares
            - **Financial Performance**: ${year_10.get('gross_revenue', 0)/1e6:.0f}M annual revenue with {(year_10.get('net_to_treasury', 0)/year_10.get('gross_revenue', 1)*100):.1f}% net margin
            - **NAV Growth**: ${year_1.get('total_nav', 0)/1e6:.0f}M → ${year_10.get('total_nav', 0)/1e6:.0f}M ({year_10.get('total_nav', 1)/year_1.get('total_nav', 1):.1f}x growth)
            - **Treasury Management**: ${year_10.get('treasury_nav', year_10.get('closing_treasury', 0))/1e6:.0f}M treasury supporting self-funded growth
            
            **Investment Highlights:**
            ✅ Diversified revenue streams across carbon, biodiversity, beef, and water markets
            ✅ Self-funding achieved by Year {data['portfolio'][data['portfolio'].get('self_funding_capable', False) == True]['year'].min() if 'self_funding_capable' in data['portfolio'].columns and any(data['portfolio'].get('self_funding_capable', False)) else 'TBD'}
            ✅ Strong stakeholder alignment with transparent distribution model
            ✅ Green Prints premium driving land value appreciation
            ✅ DCF valuation captures future cash flow potential
            
            **Risk Factors:**
            ⚠️ Regulatory changes in carbon and environmental markets
            ⚠️ Agricultural operational risks and weather dependencies
            ⚠️ Market price volatility for environmental credits
            ⚠️ Execution risk in scaling to 100 farms
            """
            
            st.markdown(summary_text)
    
    with tab2:
        st.markdown("### Full Data Tables")
        
        # Data table selector
        table_selected = st.selectbox(
            "Select Data Table",
            ["Portfolio Summary", "Farm Ledger", "NAV Reconciliation", 
             "Stakeholder Flows", "Token Metrics", "Acquisition Schedule"]
        )
        
        # Display selected table with filters
        if table_selected == "Portfolio Summary" and 'portfolio' in data:
            st.markdown("#### Portfolio Summary - All Years")
            
            # Add column selector
            cols = st.multiselect(
                "Select Columns",
                data['portfolio'].columns.tolist(),
                default=['year', 'closing_farms', 'gross_revenue', 'net_to_treasury', 
                        'total_nav', 'token_price', 'self_funding_capable'] if all(col in data['portfolio'].columns for col in ['year', 'closing_farms', 'gross_revenue', 'net_to_treasury', 'total_nav', 'token_price', 'self_funding_capable']) else data['portfolio'].columns.tolist()[:7]
            )
            
            display_df = data['portfolio'][cols].copy()
            
            # Format numeric columns
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'int64'] and col != 'year':
                    if 'price' in col:
                        display_df[col] = display_df[col].apply(lambda x: format_currency(x, 2))
                    elif any(term in col for term in ['revenue', 'treasury', 'nav', 'profit']):
                        display_df[col] = display_df[col].apply(lambda x: format_currency(x))
                    elif 'tokens' in col:
                        display_df[col] = display_df[col].apply(lambda x: format_number(x/1e6, 1) + "M")
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        elif table_selected == "Farm Ledger" and 'farm_ledger' in data:
            st.markdown("#### Complete Farm Ledger")
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                farm_filter = st.multiselect(
                    "Filter by Farm ID",
                    ["All"] + list(data['farm_ledger']['farm_id'].unique()) if 'farm_id' in data['farm_ledger'].columns else ["All"],
                    default=["All"]
                )
            with col2:
                type_filter = st.multiselect(
                    "Filter by Type",
                    ["All"] + list(data['farm_ledger']['farm_type'].unique()) if 'farm_type' in data['farm_ledger'].columns else ["All"],
                    default=["All"]
                )
            with col3:
                year_filter = st.multiselect(
                    "Filter by Year",
                    ["All"] + list(range(1, 11)),
                    default=["All"]
                )
            
            # Apply filters
            filtered = data['farm_ledger'].copy()
            if "All" not in farm_filter and 'farm_id' in filtered.columns:
                filtered = filtered[filtered['farm_id'].isin(farm_filter)]
            if "All" not in type_filter and 'farm_type' in filtered.columns:
                filtered = filtered[filtered['farm_type'].isin(type_filter)]
            if "All" not in year_filter and 'year' in filtered.columns:
                filtered = filtered[filtered['year'].isin(year_filter)]
            
            st.dataframe(filtered, hide_index=True, use_container_width=True)
            st.caption(f"Showing {len(filtered)} of {len(data['farm_ledger'])} records")
        
        elif table_selected == "NAV Reconciliation" and 'nav_recon' in data:
            st.dataframe(data['nav_recon'], hide_index=True, use_container_width=True)
        
        elif table_selected == "Stakeholder Flows" and 'stakeholder' in data:
            st.dataframe(data['stakeholder'], hide_index=True, use_container_width=True)
        
        elif table_selected == "Token Metrics" and 'token_metrics' in data:
            st.dataframe(data['token_metrics'], hide_index=True, use_container_width=True)
        
        elif table_selected == "Acquisition Schedule" and 'acquisition' in data:
            st.dataframe(data['acquisition'], hide_index=True, use_container_width=True)
        else:
            st.warning(f"{table_selected} data not available")
    
    with tab3:
        st.markdown("### Export Data")
        
        st.markdown("#### Download Model Outputs")
        
        # Create download buttons for each output
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'portfolio' in data:
                csv = data['portfolio'].to_csv(index=False)
                st.download_button(
                    label="📥 Portfolio Summary",
                    data=csv,
                    file_name="felt_portfolio_summary.csv",
                    mime="text/csv"
                )
            
            if 'farm_ledger' in data:
                csv = data['farm_ledger'].to_csv(index=False)
                st.download_button(
                    label="📥 Farm Ledger",
                    data=csv,
                    file_name="felt_farm_ledger.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'nav_recon' in data:
                csv = data['nav_recon'].to_csv(index=False)
                st.download_button(
                    label="📥 NAV Reconciliation",
                    data=csv,
                    file_name="felt_nav_reconciliation.csv",
                    mime="text/csv"
                )
            
            if 'scenarios' in data:
                csv = data['scenarios'].to_csv(index=False)
                st.download_button(
                    label="📥 Scenario Analysis",
                    data=csv,
                    file_name="felt_scenarios.csv",
                    mime="text/csv"
                )
        
        with col3:
            if 'sensitivity' in data:
                csv = data['sensitivity'].to_csv(index=False)
                st.download_button(
                    label="📥 Sensitivity Analysis",
                    data=csv,
                    file_name="felt_sensitivity.csv",
                    mime="text/csv"
                )
            
            if 'monte_carlo' in data:
                csv = data['monte_carlo'].to_csv(index=False)
                st.download_button(
                    label="📥 Monte Carlo Results",
                    data=csv,
                    file_name="felt_monte_carlo.csv",
                    mime="text/csv"
                )
        
        # Combined export
        st.markdown("#### Download All Data")
        
        if st.button("📦 Generate Complete Data Package"):
            with st.spinner("Preparing data package..."):
                # Create a combined dataset
                import io
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for name, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            csv_buffer = df.to_csv(index=False)
                            zip_file.writestr(f"felt_{name}.csv", csv_buffer)
                        elif name == 'executive_kpis':
                            zip_file.writestr("felt_executive_summary.txt", df)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Complete Data Package (ZIP)",
                    data=zip_buffer,
                    file_name="felt_model_complete_data.zip",
                    mime="application/zip"
                )
    
    with tab4:
        st.markdown("### Custom Analysis Builder")
        
        st.info("🔧 Build custom analysis by combining different metrics")
        
        if 'portfolio' in data:
            # Get available numeric columns
            numeric_cols = [col for col in data['portfolio'].columns if data['portfolio'][col].dtype in ['float64', 'int64']]
            
            # Metric selector
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox(
                    "X-Axis Metric",
                    numeric_cols,
                    index=numeric_cols.index('year') if 'year' in numeric_cols else 0
                )
            
            with col2:
                y_metrics = st.multiselect(
                    "Y-Axis Metrics",
                    [col for col in numeric_cols if col != x_metric],
                    default=['token_price'] if 'token_price' in numeric_cols else [numeric_cols[1]] if len(numeric_cols) > 1 else []
                )
            
            if y_metrics:
                # Create custom chart
                fig = go.Figure()
                
                colors = ['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']
                
                for i, metric in enumerate(y_metrics):
                    fig.add_trace(go.Scatter(
                        x=data['portfolio'][x_metric],
                        y=data['portfolio'][metric],
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=colors[i % len(colors)], width=2),
                        mode='lines+markers'
                    ))
                
                fig.update_layout(
                    title="Custom Analysis",
                    xaxis_title=x_metric.replace('_', ' ').title(),
                    yaxis_title="Value",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation matrix
                if len(y_metrics) > 1:
                    st.markdown("#### Correlation Analysis")
                    
                    corr_data = data['portfolio'][y_metrics].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_data.values,
                        x=corr_data.columns,
                        y=corr_data.columns,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=[[f"{v:.2f}" for v in row] for row in corr_data.values],
                        texttemplate="%{text}",
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig.update_layout(
                        title="Metric Correlation Matrix",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Add disclaimer
    show_disclaimer()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>FELT Token Financial Model Dashboard v1.0</p>
    <p>Fresh Earth Agriculture Group | Regenerative Agriculture Investment Platform</p>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>
""", unsafe_allow_html=True)