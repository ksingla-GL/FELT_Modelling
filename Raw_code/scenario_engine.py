"""
FELT Token - Scenario & Sensitivity Engine
Complete version with all fixes
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path adjustment
from core_model import FELTModel

class ScenarioEngine:
    """Scenario and sensitivity analysis for FELT model"""
    
    def __init__(self):
        self.base_model = FELTModel()
        
    def modify_model(self, model, adjustments):
        """Apply adjustments to model parameters with proper propagation"""
        for param, value in adjustments.items():
            if '.' in param:
                # Nested parameter (e.g., 'prices.forestry')
                parts = param.split('.')
                if parts[0] == 'prices':
                    if parts[1] in model.base_prices:
                        model.base_prices[parts[1]] = value
                        # Recalculate all year prices
                        for year in range(1, 11):
                            price = value * (1 + model.price_growth[parts[1]]) ** (year - 1)
                            model.prices[(parts[1], year)] = price
                elif parts[0] == 'costs':
                    if parts[1] in model.costs:
                        old_val = model.costs[parts[1]]
                        model.costs[parts[1]] = value
                        # Rebalance treasury if needed
                        if parts[1] != 'treasury_pretax':
                            model.costs['treasury_pretax'] += (old_val - value)
            elif param == 'revenue_scale':
                # Scale all revenue volumes
                for key in model.revenue_lookup:
                    model.revenue_lookup[key] *= value
            elif param in model.portfolio:
                model.portfolio[param] = value
                
        # CRITICAL: Propagate changes to derived parameters
        model.caps['land_appreciation_rate'] = float(model.portfolio.get('land_appreciation_rate', 
                                                                        model.caps['land_appreciation_rate']))
        model.caps['operator_salary_cap'] = float(model.portfolio.get('operator_salary_cap',
                                                                     model.caps['operator_salary_cap']))
        model.green_prints['max_premium'] = float(model.portfolio.get('green_prints_max_premium',
                                                                     model.green_prints['max_premium']))
        model.green_prints['start_age'] = int(model.portfolio.get('green_prints_start_age',
                                                                 model.green_prints['start_age']))
        model.green_prints['ramp_years'] = int(model.portfolio.get('green_prints_ramp_years',
                                                                  model.green_prints['ramp_years']))
                
        return model
        
    def run_scenarios(self):
        """Run three main scenarios"""
        scenarios = {
            'Conservative': {
                'prices.forestry': 35,      # Lower ACCU price
                'prices.soil': 35,           # Lower ACCU price
                'prices.biodiversity': 9000, # Lower biodiversity
                'prices.beef': 3500,         # Lower beef price
                'prices.water': 4000,        # Lower water price
                'revenue_scale': 0.8,        # 80% of base volumes
                'land_appreciation_rate': 0.03,  # 3% appreciation
                'green_prints_max_premium': 0.10 # 10% max premium
            },
            'Base': {},  # Use defaults from inputs
            'Aggressive': {
                'prices.forestry': 55,       # Higher ACCU price
                'prices.soil': 55,           # Higher ACCU price
                'prices.biodiversity': 15000, # Higher biodiversity
                'prices.beef': 4500,         # Premium beef price
                'prices.water': 6000,        # Higher water price
                'revenue_scale': 1.2,        # 120% of base volumes
                'land_appreciation_rate': 0.05,  # 5% appreciation
                'green_prints_max_premium': 0.20 # 20% max premium
            }
        }
        
        results_summary = []
        all_results = []
        
        for scenario_name, adjustments in scenarios.items():
            print(f"  Running {scenario_name} scenario...")
            
            # Create fresh model instance
            model = FELTModel()
            model = self.modify_model(model, adjustments)
            
            # Run projection
            df = model.run()
            df['scenario'] = scenario_name
            all_results.append(df)
            
            # Extract summary metrics
            year_10 = df.iloc[-1]
            self_funding = df[df['self_funding_capable']]['year'].min() if any(df['self_funding_capable']) else 0
            
            # Get mature farm profit
            mature_farms = [f for f in model.farms if len(f.ledger) >= 10]
            min_profit = min([f.cum_profit for f in mature_farms]) if mature_farms else 0
            
            results_summary.append({
                'scenario': scenario_name,
                'year_10_token_price': year_10['token_price'],
                'token_multiple': year_10['token_price'],
                'year_10_nav': year_10['total_nav'],
                'year_10_revenue': year_10['gross_revenue'],
                'total_pdf': df['pdf_total'].sum(),
                'min_farm_profit': min_profit,
                'self_funding_year': self_funding,
                'year_10_treasury': year_10['treasury_nav'],
                'year_10_land': year_10['land_nav']
            })
            
        # Save results
        pd.DataFrame(results_summary).to_csv('outputs/scenario_comparison.csv', index=False)
        
        # Save detailed results
        combined_df = pd.concat(all_results)
        combined_df.to_csv('outputs/scenario_details.csv', index=False)
        
        return pd.DataFrame(results_summary)
        
    def run_sensitivity(self):
        """Run sensitivity analysis on key parameters"""
        
        sensitivity_params = {
            'ACCU Price': {
                'param': 'prices.forestry',
                'values': [30, 35, 40, 45, 50, 55, 60],
                'linked': {'prices.soil': 1.0}  # Soil price tracks forestry
            },
            'PDF Rate': {
                'param': 'costs.project_development',
                'values': [0.10, 0.125, 0.15, 0.175, 0.20],
                'rebalance': True
            },
            'Treasury Rate': {
                'param': 'costs.treasury_pretax',
                'values': [0.25, 0.28, 0.32, 0.36, 0.40],
                'rebalance': False  # Don't rebalance, let total go over 100%
            },
            'Land Appreciation': {
                'param': 'land_appreciation_rate',
                'values': [0.02, 0.03, 0.04, 0.05, 0.06]
            },
            'Operator Cap': {
                'param': 'operator_salary_cap',
                'values': [200000, 225000, 250000, 275000, 300000]
            },
            'Revenue Volume': {
                'param': 'revenue_scale',
                'values': [0.6, 0.8, 1.0, 1.2, 1.4]
            },
            'Tax Rate': {
                'param': 'corporate_tax_rate',
                'values': [0.20, 0.25, 0.30, 0.35, 0.40]
            }
        }
        
        results = []
        
        for param_name, config in sensitivity_params.items():
            print(f"  Testing sensitivity: {param_name}")
            
            for value in config['values']:
                # Fresh model
                model = FELTModel()
                
                # Apply main adjustment
                adjustments = {config['param']: value}
                
                # Apply linked adjustments
                if 'linked' in config:
                    for linked_param, multiplier in config['linked'].items():
                        adjustments[linked_param] = value * multiplier
                        
                model = self.modify_model(model, adjustments)
                
                # Run and extract metrics
                df = model.run()
                year_5 = df.iloc[4] if len(df) > 4 else df.iloc[-1]
                year_10 = df.iloc[-1]
                
                # Get mature farm profit
                mature_farms = [f for f in model.farms if len(f.ledger) >= 10]
                min_profit = min([f.cum_profit for f in mature_farms]) if mature_farms else 0
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'year_5_token_price': year_5['token_price'],
                    'year_10_token_price': year_10['token_price'],
                    'year_10_nav': year_10['total_nav'],
                    'year_10_revenue': year_10['gross_revenue'],
                    'total_pdf': df['pdf_total'].sum(),
                    'token_multiple': year_10['token_price'],
                    'min_farm_profit': min_profit,
                    'year_10_treasury': year_10['treasury_nav'],
                    'year_10_land': year_10['land_nav']
                })
                
        sensitivity_df = pd.DataFrame(results)
        sensitivity_df.to_csv('outputs/sensitivity_analysis.csv', index=False)
        
        return sensitivity_df
    
    def run_monte_carlo(self, n_simulations=100):
        """Run Monte Carlo simulation for risk analysis"""
        
        print(f"\nRunning {n_simulations} Monte Carlo simulations...")
        
        results = []
        
        for sim in range(n_simulations):
            if sim % 20 == 0:
                print(f"  Simulation {sim+1}/{n_simulations}")
            
            # Create random adjustments within reasonable ranges
            adjustments = {
                'prices.forestry': np.random.uniform(35, 55),
                'prices.soil': np.random.uniform(35, 55),
                'prices.biodiversity': np.random.uniform(9000, 15000),
                'prices.beef': np.random.uniform(3500, 4500),
                'prices.water': np.random.uniform(4000, 6000),
                'revenue_scale': np.random.uniform(0.8, 1.2),
                'land_appreciation_rate': np.random.uniform(0.03, 0.05),
                'corporate_tax_rate': np.random.uniform(0.25, 0.35)
            }
            
            # Run model with random parameters
            model = FELTModel()
            model = self.modify_model(model, adjustments)
            df = model.run()
            
            year_10 = df.iloc[-1]
            
            results.append({
                'simulation': sim + 1,
                'token_price': year_10['token_price'],
                'nav': year_10['total_nav'],
                'self_funding': df[df['self_funding_capable']]['year'].min() if any(df['self_funding_capable']) else 0
            })
        
        monte_carlo_df = pd.DataFrame(results)
        
        # Calculate statistics
        stats = {
            'metric': ['Token Price', 'NAV', 'Self-Funding Year'],
            'mean': [
                monte_carlo_df['token_price'].mean(),
                monte_carlo_df['nav'].mean(),
                monte_carlo_df['self_funding'].mean()
            ],
            'std': [
                monte_carlo_df['token_price'].std(),
                monte_carlo_df['nav'].std(),
                monte_carlo_df['self_funding'].std()
            ],
            'min': [
                monte_carlo_df['token_price'].min(),
                monte_carlo_df['nav'].min(),
                monte_carlo_df['self_funding'].min()
            ],
            'p25': [
                monte_carlo_df['token_price'].quantile(0.25),
                monte_carlo_df['nav'].quantile(0.25),
                monte_carlo_df['self_funding'].quantile(0.25)
            ],
            'median': [
                monte_carlo_df['token_price'].median(),
                monte_carlo_df['nav'].median(),
                monte_carlo_df['self_funding'].median()
            ],
            'p75': [
                monte_carlo_df['token_price'].quantile(0.75),
                monte_carlo_df['nav'].quantile(0.75),
                monte_carlo_df['self_funding'].quantile(0.75)
            ],
            'max': [
                monte_carlo_df['token_price'].max(),
                monte_carlo_df['nav'].max(),
                monte_carlo_df['self_funding'].max()
            ]
        }
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv('outputs/monte_carlo_stats.csv', index=False)
        monte_carlo_df.to_csv('outputs/monte_carlo_results.csv', index=False)
        
        return stats_df

if __name__ == "__main__":
    print("=" * 60)
    print("SCENARIO & SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    engine = ScenarioEngine()
    
    print("\nRunning scenarios...")
    scenarios = engine.run_scenarios()
    
    print("\nScenario Results:")
    print(scenarios[['scenario', 'token_multiple', 'min_farm_profit', 'self_funding_year']])
    
    print("\nRunning sensitivity analysis...")
    sensitivity = engine.run_sensitivity()
    
    print("\nSensitivity Ranges:")
    for param in sensitivity['parameter'].unique():
        param_data = sensitivity[sensitivity['parameter'] == param]
        print(f"  {param}:")
        print(f"    Token Price: ${param_data['year_10_token_price'].min():.2f} - ${param_data['year_10_token_price'].max():.2f}")
        print(f"    Multiple: {param_data['token_multiple'].min():.1f}x - {param_data['token_multiple'].max():.1f}x")
        print(f"    Farm Profit: ${param_data['min_farm_profit'].min():,.0f} - ${param_data['min_farm_profit'].max():,.0f}")
    
    # Optional: Run Monte Carlo
    print("\nRun Monte Carlo simulation? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        stats = engine.run_monte_carlo(100)
        print("\nMonte Carlo Statistics:")
        print(stats)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutputs generated:")
    print("  - scenario_comparison.csv")
    print("  - scenario_details.csv")
    print("  - sensitivity_analysis.csv")
    if response == 'y':
        print("  - monte_carlo_stats.csv")
        print("  - monte_carlo_results.csv")
    print("=" * 60)