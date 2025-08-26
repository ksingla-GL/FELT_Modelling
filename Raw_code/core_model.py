"""
FELT Token Financial Model - Core Engine (FIXED)
All treasury and NAV issues resolved
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

STREAMS = ['forestry', 'soil', 'biodiversity', 'beef']

class CoP:
    """Change of Practice (farm) entity"""
    
    def __init__(self, cop_id, cop_type, acquisition_year, hectares, land_cost, dev_cost):
        self.id = cop_id
        self.type = cop_type
        self.a_year = acquisition_year
        self.hectares = hectares
        self.land_fmv = land_cost + dev_cost
        self.initial_cost = land_cost + dev_cost
        self.cum_profit = 0.0
        self.ledger = []
        
    def compute_year(self, year, prices, stream_costs, caps, tax_rate, revenue_lookup, green_prints_params):
        """Compute farm P&L for given year"""
        age = year - self.a_year + 1
        if age < 1 or age > 10:
            return None
            
        # Gross revenue and stream-level costs
        gross = 0.0
        streams = {}
        stream_profits = {}
        total_pays = {'project_development': 0, 'expert': 0, 'supplier': 0, 'admin': 0, 'operator': 0}
        total_treasury_pretax = 0.0
        
        for stream in STREAMS:
            units_per_ha = revenue_lookup.get((self.type, stream, age), 0)
            units = units_per_ha * self.hectares
            price = prices.get((stream, year), 0)
            stream_rev = units * price
            streams[stream] = stream_rev
            gross += stream_rev
            
            # Stream-level cost allocations
            stream_cost_rates = stream_costs[stream]
            stream_pays = {k: stream_cost_rates[k] * stream_rev for k in stream_cost_rates.keys()}
            
            # Accumulate totals for each stakeholder
            for stakeholder in ['project_development', 'expert', 'supplier', 'admin', 'operator']:
                total_pays[stakeholder] += stream_pays.get(stakeholder, 0)
            
            # Stream treasury (before operator cap adjustment)
            stream_treasury = stream_pays.get('treasury_pretax', 0)
            total_treasury_pretax += stream_treasury
            
            # Store stream profit (before operator cap and tax)
            stream_profits[stream] = stream_treasury
        
        # Operator cap adjustment
        op_paid = min(total_pays['operator'], caps['operator_salary_cap'])
        op_over = max(total_pays['operator'] - op_paid, 0.0)
        
        # Adjusted treasury (add operator overflow)
        t_pre = total_treasury_pretax + op_over
        
        # Tax & net-to-treasury
        tax = tax_rate * t_pre
        ntt = t_pre - tax
        self.cum_profit += ntt
        
        # Land value update - FIXED Green Prints logic
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
            'pdf_fee': total_pays.get('project_development', 0),
            'expert_fee': total_pays.get('expert', 0),
            'supplier_costs': total_pays.get('supplier', 0),
            'admin': total_pays.get('admin', 0),
            'operator_paid': op_paid,
            'operator_overflow': op_over,
            'treasury_pretax': t_pre,
            'tax': tax,
            'net_to_treasury': ntt,
            'cum_profit': self.cum_profit,
            'land_fmv': self.land_fmv,
            'forestry_profit': stream_profits.get('forestry', 0),
            'soil_profit': stream_profits.get('soil', 0),
            'biodiversity_profit': stream_profits.get('biodiversity', 0),
            'beef_profit': stream_profits.get('beef', 0)
        }
        
        self.ledger.append(row)
        return row

class FELTModel:
    """Main FELT token financial model with FIXED treasury handling"""
    
    def __init__(self):
        self.load_inputs()
        self.farms = []
        self.treasury = 0.0  # START AT ZERO - CRITICAL FIX
        self.tokens_outstanding = 0.0
        self.farm_counter = 0
        
    def load_inputs(self):
        """Load all CSV inputs with proper type conversion"""
        # Portfolio config
        portfolio_df = pd.read_csv('inputs/1_portfolio_config.csv')
        self.portfolio = {}
        for _, row in portfolio_df.iterrows():
            param = row['parameter']
            value = row['value']
            # Convert numeric parameters to float
            if param in ['land_cost_per_farm', 'dev_cost_per_farm', 'hectares_per_farm',
                        'land_appreciation_rate', 'operator_salary_cap', 'corporate_tax_rate',
                        'working_capital_reserve_pct', 'initial_token_supply', 'initial_raise',
                        'issuance_premium', 'green_prints_start_age', 'green_prints_max_premium',
                        'green_prints_ramp_years']:
                self.portfolio[param] = float(value)
            else:
                self.portfolio[param] = value
        
        # Acquisition mix
        self.acquisition_mix = pd.read_csv('inputs/2_acquisition_mix.csv')
        
        # Revenue timelines
        revenue_df = pd.read_csv('inputs/3_cop_revenue_timelines.csv')
        self.revenue_lookup = {}
        for _, row in revenue_df.iterrows():
            key = (row['cop_type'], row['stream'], int(row['operational_age']))
            self.revenue_lookup[key] = float(row['units_per_hectare'])
        
        # Cost structure - stream-based
        cost_df = pd.read_csv('inputs/4_cost_structure.csv')
        self.stream_costs = {}
        for _, row in cost_df.iterrows():
            stream = row['stream']
            stakeholder = row['stakeholder']
            if stream not in self.stream_costs:
                self.stream_costs[stream] = {}
            self.stream_costs[stream][stakeholder] = float(row['percentage'])
        
        # Calculate treasury rates as balancing figures for each stream
        for stream in self.stream_costs:
            total_other_costs = sum(self.stream_costs[stream].values())
            self.stream_costs[stream]['treasury_pretax'] = 1.0 - total_other_costs
        
        # Market prices
        prices_df = pd.read_csv('inputs/5_market_prices.csv')
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
        
    def acquire_batch(self, year):
        """Acquire farms with CORRECT treasury handling"""
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
        
        # CRITICAL FIX: Proper capital and treasury handling
        if year == 1:
            # Year 1: Initial raise
            capital_raised = self.portfolio['initial_raise']
            new_tokens = self.portfolio['initial_token_supply']
            self.tokens_outstanding = new_tokens
            
            # Add capital to treasury
            self.treasury += capital_raised
            
            # PAY FOR FARMS FROM TREASURY - CRITICAL
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
        
        # Create farms
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
        
    def calculate_nav(self):
        """Calculate total NAV = Land + Treasury"""
        land_nav = sum(farm.land_fmv for farm in self.farms)
        treasury_nav = self.treasury
        return land_nav + treasury_nav
        
    
    def calculate_dcf_token_value(self, year, current_nav, current_treasury):
        """Calculate token value using DCF of future cash flows"""
        
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
        # Use recent treasury growth as proxy for cash generation
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

    
    def run(self):
        """Run 10-year projection with FIXED accounting"""
        results = []
        
        # DO NOT set initial treasury here - it starts at 0
        
        for year in range(1, 11):
            year_data = {
                'year': year,
                'opening_farms': len(self.farms),
                'opening_treasury': self.treasury,
                'opening_nav': 0 if year == 1 else self.calculate_nav()  # Year 1 starts at 0
            }
            
            # Acquire farms at start of year
            acquisition = self.acquire_batch(year)
            year_data['farms_acquired'] = acquisition['farms_count']
            year_data['capital_raised'] = acquisition['capital_raised']
            year_data['new_tokens'] = acquisition['new_tokens']
            year_data['funding_source'] = acquisition['funding_source']
            
            # Process all farms for the year
            farm_results = []
            for farm in self.farms:
                result = farm.compute_year(
                    year, 
                    self.prices, 
                    self.stream_costs, 
                    self.caps,
                    self.portfolio['corporate_tax_rate'],
                    self.revenue_lookup,
                    self.green_prints
                )
                if result:
                    farm_results.append(result)
                    
            # Aggregate results
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
                
                # Add to treasury
                self.treasury += year_data['net_to_treasury']
            else:
                for key in ['gross_revenue', 'pdf_total', 'expert_total', 'supplier_total',
                          'admin_total', 'operator_paid_total', 'operator_overflow_total',
                          'treasury_pretax_total', 'tax_total', 'net_to_treasury']:
                    year_data[key] = 0
                    
            # End of year metrics
            year_data['closing_farms'] = len(self.farms)
            year_data['closing_treasury'] = self.treasury
            year_data['land_nav'] = sum(farm.land_fmv for farm in self.farms)
            year_data['treasury_nav'] = self.treasury
            year_data['total_nav'] = year_data['land_nav'] + year_data['treasury_nav']
            year_data['tokens_outstanding'] = self.tokens_outstanding
            # Calculate both standard and DCF token prices
            standard_token_price = (year_data['total_nav'] / self.tokens_outstanding 
                                   if self.tokens_outstanding > 0 else 1.0)
            
            # Calculate DCF-based token price
            dcf_token_price, pv_future = self.calculate_dcf_token_value(
                year, 
                year_data['total_nav'],
                year_data['treasury_nav']
            )
            
            # Use DCF price as primary token price
            year_data['token_price'] = dcf_token_price
            year_data['standard_token_price'] = standard_token_price
            year_data['pv_future_cash'] = pv_future
            year_data['dcf_premium'] = (dcf_token_price / standard_token_price - 1) if standard_token_price > 0 else 0
            
            # Check self-funding threshold
            reserve_factor = 1 - self.portfolio['working_capital_reserve_pct']
            next_year_cost = 10 * (self.portfolio['land_cost_per_farm'] + 
                                  self.portfolio['dev_cost_per_farm'])
            year_data['self_funding_capable'] = (self.treasury * reserve_factor >= next_year_cost)
            
            # Track cumulative profits
            if self.farms:
                oldest_farms = sorted(self.farms, key=lambda f: f.a_year)[:10]
                year_data['oldest_farm_cum_profit'] = oldest_farms[0].cum_profit if oldest_farms else 0
                year_data['avg_cum_profit_first_10'] = (sum(f.cum_profit for f in oldest_farms) / 
                                                        len(oldest_farms) if oldest_farms else 0)
            else:
                year_data['oldest_farm_cum_profit'] = 0
                year_data['avg_cum_profit_first_10'] = 0
                
            results.append(year_data)
            
        return pd.DataFrame(results)
        
    def generate_outputs(self, df):
        """Generate outputs with CORRECT NAV reconciliation"""
        
        # Portfolio summary
        portfolio_cols = ['year', 'opening_farms', 'farms_acquired', 'closing_farms',
                         'gross_revenue', 'net_to_treasury', 'closing_treasury',
                         'land_nav', 'treasury_nav', 'total_nav', 'token_price',
                         'standard_token_price', 'dcf_premium', 'pv_future_cash',
                         'self_funding_capable', 'oldest_farm_cum_profit']
        df[portfolio_cols].to_csv('outputs/output_portfolio_summary.csv', index=False)
        
        # Stakeholder flows
        stakeholder_cols = ['year', 'pdf_total', 'expert_total', 'supplier_total',
                          'admin_total', 'operator_paid_total', 'operator_overflow_total',
                          'tax_total']
        df[stakeholder_cols].to_csv('outputs/output_stakeholder_flows.csv', index=False)
        
        # Farm ledger
        farm_ledger = []
        for farm in self.farms:
            for entry in farm.ledger:
                row = {
                    'farm_id': farm.id,
                    'farm_type': farm.type,
                    'acquisition_year': farm.a_year,
                    **{k: v for k, v in entry.items() if k != 'streams'}
                }
                farm_ledger.append(row)
        if farm_ledger:
            pd.DataFrame(farm_ledger).to_csv('outputs/output_farm_ledger.csv', index=False)
            
        # Stream-level breakdown
        stream_breakdown = []
        for farm in self.farms:
            for entry in farm.ledger:
                farm_gross_rev = entry.get('gross', 0)
                if farm_gross_rev > 0:
                    # Calculate total uncapped operator cost for this farm
                    total_uncapped_operator = 0
                    stream_revenues = {}
                    for stream in STREAMS:
                        stream_rev = entry.get(f'gross_{stream}', 0)
                        if stream_rev > 0:
                            stream_revenues[stream] = stream_rev
                            stream_costs = self.stream_costs[stream]
                            total_uncapped_operator += stream_costs['operator'] * stream_rev
                    
                    # Apply operator cap at farm level
                    capped_operator = min(total_uncapped_operator, self.caps['operator_salary_cap'])
                    operator_overflow = max(total_uncapped_operator - capped_operator, 0.0)
                    
                    # Allocate costs across streams
                    for stream in STREAMS:
                        stream_rev = stream_revenues.get(stream, 0)
                        if stream_rev > 0:
                            stream_costs = self.stream_costs[stream]
                            
                            # Proportional allocation of capped operator cost
                            stream_share = stream_rev / farm_gross_rev
                            allocated_operator = capped_operator * stream_share
                            
                            # Calculate other costs normally
                            expert_cost = stream_costs['expert'] * stream_rev
                            supplier_cost = stream_costs['supplier'] * stream_rev
                            project_dev_cost = stream_costs['project_development'] * stream_rev
                            admin_cost = stream_costs['admin'] * stream_rev
                            
                            # Treasury gets the normal rate plus proportional operator overflow
                            base_treasury = stream_costs['treasury_pretax'] * stream_rev
                            overflow_share = operator_overflow * stream_share
                            total_treasury = base_treasury + overflow_share
                            
                            stream_row = {
                                'farm_id': farm.id,
                                'farm_type': farm.type,
                                'year': entry['year'],
                                'age': entry['age'],
                                'stream': stream,
                                'gross_revenue': stream_rev,
                                'expert_cost': expert_cost,
                                'supplier_cost': supplier_cost,
                                'operator_cost': allocated_operator,
                                'operator_overflow_allocated': overflow_share,
                                'project_dev_cost': project_dev_cost,
                                'admin_cost': admin_cost,
                                'treasury_pretax_rate': stream_costs['treasury_pretax'],
                                'treasury_pretax_base': base_treasury,
                                'treasury_pretax_total': total_treasury,
                                'net_profit_before_tax': entry.get(f'{stream}_profit', 0)
                            }
                            stream_breakdown.append(stream_row)
        
        if stream_breakdown:
            pd.DataFrame(stream_breakdown).to_csv('outputs/output_stream_breakdown.csv', index=False)
        
        # Acquisition schedule
        acq_schedule = df[['year', 'farms_acquired', 'capital_raised', 'new_tokens',
                         'funding_source']].copy()
        acq_schedule.to_csv('outputs/output_acquisition_schedule.csv', index=False)
        
        # NAV reconciliation - FIXED to not include capital
        nav_recon = []
        cost_per_farm = self.portfolio['land_cost_per_farm'] + self.portfolio['dev_cost_per_farm']
        
        for i, row in df.iterrows():
            if i == 0:
                # Year 1 - Start from zero
                opening_nav = 0
                new_capital = row['capital_raised']  # Include capital in NAV
                
                # Land appreciation only
                land_at_cost = row['farms_acquired'] * cost_per_farm
                land_total = row['land_nav']
                land_appreciation = land_total - land_at_cost
                
                operating = row['net_to_treasury']
                closing = row['total_nav']
                
                recon = {
                    'year': row['year'],
                    'opening_nav': 0,
                    'new_capital': new_capital,  # Capital increases NAV
                    'land_revaluation': land_appreciation,
                    'operating_result': operating,
                    'closing_nav': closing
                }
            else:
                # Years 2-10
                prev_row = df.iloc[i-1]
                opening_nav = prev_row['total_nav']
                new_capital = row['capital_raised']  # Include capital in NAV
                
                # Land appreciation only
                land_change = row['land_nav'] - prev_row['land_nav']
                new_farms_cost = row['farms_acquired'] * cost_per_farm
                land_appreciation = land_change - new_farms_cost
                
                operating = row['net_to_treasury']
                closing = row['total_nav']
                
                recon = {
                    'year': row['year'],
                    'opening_nav': opening_nav,
                    'new_capital': new_capital,  # Capital increases NAV
                    'land_revaluation': land_appreciation,
                    'operating_result': operating,
                    'closing_nav': closing
                }
            
            nav_recon.append(recon)
        
        nav_df = pd.DataFrame(nav_recon)
        
        # Verify with CORRECT formula (includes capital)
        nav_df['calculated'] = (nav_df['opening_nav'] + 
                               nav_df['new_capital'] +  # Include capital
                               nav_df['land_revaluation'] + 
                               nav_df['operating_result'])
        nav_df['check'] = abs(nav_df['calculated'] - nav_df['closing_nav']) < 1
        
        if not nav_df['check'].all():
            print('NAV Reconciliation Check:')
            print('Max discrepancy: ${:,.0f}'.format(
                abs(nav_df['calculated'] - nav_df['closing_nav']).max()))
        
        nav_df[['year', 'opening_nav', 'new_capital', 'land_revaluation',
                'operating_result', 'closing_nav']].to_csv(
                'outputs/output_nav_reconciliation.csv', index=False)

        
        # Token metrics
        token_metrics = df[['year', 'total_nav', 'tokens_outstanding', 'token_price']].copy()
        token_metrics['token_multiple'] = token_metrics['token_price'] / 1.0
        token_metrics.to_csv('outputs/output_token_metrics.csv', index=False)
        
        # Executive KPIs
        year_10 = df.iloc[-1]
        self_funding_year = df[df['self_funding_capable']]['year'].min() if any(df['self_funding_capable']) else 'Not achieved'
        
        mature_farms = [f for f in self.farms if len(f.ledger) >= 10]
        min_profit = min([f.cum_profit for f in mature_farms]) if mature_farms else 0
        
        kpis = f"""FELT Token Model - Executive KPIs
=====================================
Year 10 Token Price (DCF): ${year_10['token_price']:.2f}
Year 10 Token Price (NAV): ${year_10.get('standard_token_price', year_10['token_price']):.2f}
DCF Premium: {year_10.get('dcf_premium', 0)*100:.1f}%
Token Multiple (DCF): {year_10['token_price']:.2f}x
Token Multiple (NAV): {year_10.get('standard_token_price', year_10['token_price']):.2f}x
Self-Funding Year: {self_funding_year}
Total Farms: {year_10['closing_farms']}
Portfolio NAV: ${year_10['total_nav']:,.0f}
Land NAV: ${year_10['land_nav']:,.0f}
Treasury NAV: ${year_10['treasury_nav']:,.0f}
Total FEAG PDF: ${df['pdf_total'].sum():,.0f}
Min Mature Farm Profit: ${min_profit:,.0f}

Key Checks:
- Year 1 Treasury after purchases: ${df.iloc[0]['closing_treasury']:,.0f}
- Year 1 NAV: ${df.iloc[0]['total_nav']:,.0f} (should be ~$140M + operations)
- NAV Formula: Land + Treasury (no double counting)
"""
        
        with open('outputs/output_executive_kpi.txt', 'w') as f:
            f.write(kpis)
            
        return kpis

if __name__ == "__main__":
    print("=" * 60)
    print("FELT MODEL - FIXED VERSION")
    print("=" * 60)
    
    model = FELTModel()
    results = model.run()
    kpis = model.generate_outputs(results)
    
    print(kpis)
    print("All outputs generated in /outputs/ directory")
