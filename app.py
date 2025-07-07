import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Set page config
st.set_page_config(page_title="GammaX", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 0rem;
        background-color: #000000;
    }
    
    /* Header styling */
    .main-header {
        background: #000000;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: black;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #000000;
        color: white !important;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #302e2e;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: #302e2e;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 2px solid #333333;
    }
    
    /* Ensure tab text is always white */
    .stTabs [data-baseweb="tab"] span {
        color: white !important;
    }
    
    /* Metric container styling */
    div[data-testid="metric-container"] {
        background-color: #302e2e;
        border: 1px solid #e1e4e8;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: #302e2e !important;
        border: 1px solid #d95c1e;
        color: white !important;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #333333 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #000000 !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Developer credit styling */
    .developer-credit {
        text-align: center;
        padding: 2rem;
        background: #000000;
        color: white;
        border-radius: 10px;
        margin-top: 3rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .developer-credit a {
        color: #64b5f6;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .developer-credit a:hover {
        color: #90caf9;
        text-decoration: underline;
    }
    
    /* Card styling */
    .info-card {
        background: black;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Expander button styling */
    .streamlit-expanderHeader {
        background-color: #000000 !important;
        color: white !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 2.5rem;">GammaX</h1>
        <p style="margin:0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Professional-grade option pricing with multiple models, Greeks analysis, and strategy builder
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for global parameters
with st.sidebar:
    st.markdown("### Global Parameters")
    
    spot_price = st.number_input("Spot Price ($)", value=100.0, min_value=0.01, step=0.01)
    strike_price = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=0.25, min_value=0.001, max_value=5.0, step=0.01)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1) / 100
    volatility = st.number_input("Volatility (%)", value=20.0, min_value=0.1, max_value=200.0, step=0.1) / 100
    dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=20.0, step=0.1) / 100
    
    st.markdown("### Model-Specific Parameters")
    
    # Binomial Tree parameters
    binomial_steps = st.number_input("Binomial Tree Steps", value=100, min_value=10, max_value=1000, step=10)
    
    # Monte Carlo parameters
    mc_simulations = st.number_input("Monte Carlo Simulations", value=10000, min_value=1000, max_value=100000, step=1000)
    
    # Heston model parameters
    with st.expander("Heston Model Parameters"):
        kappa = st.number_input("Mean Reversion Speed (κ)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
        theta = st.number_input("Long-term Variance (θ)", value=0.04, min_value=0.01, max_value=1.0, step=0.01)
        sigma_v = st.number_input("Vol of Vol (σᵥ)", value=0.3, min_value=0.01, max_value=2.0, step=0.01)
        rho_heston = st.number_input("Correlation (ρ)", value=-0.7, min_value=-1.0, max_value=1.0, step=0.1)
        v0 = st.number_input("Initial Variance (v₀)", value=0.04, min_value=0.01, max_value=1.0, step=0.01)
    
    # Merton Jump Diffusion parameters
    with st.expander("Merton Jump Model Parameters"):
        lambda_jump = st.number_input("Jump Intensity (λ)", value=0.1, min_value=0.0, max_value=2.0, step=0.01)
        mu_jump = st.number_input("Mean Jump Size (μⱼ)", value=-0.1, min_value=-1.0, max_value=1.0, step=0.01)
        sigma_jump = st.number_input("Jump Volatility (σⱼ)", value=0.15, min_value=0.01, max_value=1.0, step=0.01)

# Option Pricing Models
class OptionPricingModels:
    
    @staticmethod
    def black_scholes(S, K, T, r, sigma, q, option_type='call'):
        """Black-Scholes option pricing model"""
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def binomial_tree(S, K, T, r, sigma, q, n, option_type='call', american=False):
        """Binomial tree option pricing model"""
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(n + 1)
        for i in range(n + 1):
            asset_prices[i] = S * (u ** (n - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(n + 1)
        for i in range(n + 1):
            if option_type == 'call':
                option_values[i] = max(0, asset_prices[i] - K)
            else:
                option_values[i] = max(0, K - asset_prices[i])
        
        # Step back through the tree
        for j in range(n - 1, -1, -1):
            for i in range(j + 1):
                asset_price = S * (u ** (j - i)) * (d ** i)
                option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                if american:
                    if option_type == 'call':
                        option_values[i] = max(option_values[i], asset_price - K)
                    else:
                        option_values[i] = max(option_values[i], K - asset_price)
        
        return option_values[0]
    
    @staticmethod
    def monte_carlo(S, K, T, r, sigma, q, n_sims, option_type='call'):
        """Monte Carlo simulation for option pricing"""
        np.random.seed(42)
        Z = np.random.standard_normal(n_sims)
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_sims)
        
        return price, std_error
    
    @staticmethod
    def heston_monte_carlo(S, K, T, r, q, v0, kappa, theta, sigma_v, rho, n_sims, n_steps, option_type='call'):
        """Heston stochastic volatility model using Monte Carlo"""
        np.random.seed(42)
        dt = T / n_steps
        
        # Initialize arrays
        S_paths = np.zeros((n_sims, n_steps + 1))
        v_paths = np.zeros((n_sims, n_steps + 1))
        S_paths[:, 0] = S
        v_paths[:, 0] = v0
        
        # Generate correlated random numbers
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_sims)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_sims)
            
            # Variance process (CIR)
            v_paths[:, i + 1] = v_paths[:, i] + kappa * (theta - v_paths[:, i]) * dt + \
                                sigma_v * np.sqrt(np.maximum(v_paths[:, i], 0)) * np.sqrt(dt) * Z2
            v_paths[:, i + 1] = np.maximum(v_paths[:, i + 1], 0)
            
            # Stock price process
            S_paths[:, i + 1] = S_paths[:, i] * np.exp((r - q - 0.5 * v_paths[:, i]) * dt + 
                                                       np.sqrt(v_paths[:, i]) * np.sqrt(dt) * Z1)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_sims)
        
        return price, std_error, S_paths
    
    @staticmethod
    def merton_jump_diffusion(S, K, T, r, sigma, q, lambda_j, mu_j, sigma_j, n_terms, option_type='call'):
        """Merton jump-diffusion model"""
        price = 0
        
        for n in range(n_terms):
            # Adjusted parameters for n jumps
            r_n = r - lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1) + n * (mu_j + 0.5 * sigma_j**2) / T
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
            
            # Poisson probability
            poisson_prob = np.exp(-lambda_j * T) * (lambda_j * T)**n / math.factorial(n)
            
            # Black-Scholes price with adjusted parameters
            bs_price = OptionPricingModels.black_scholes(S, K, T, r_n, sigma_n, q, option_type)
            
            price += poisson_prob * bs_price
        
        return price

# Greeks Calculation
class Greeks:
    
    @staticmethod
    def calculate_all_greeks(S, K, T, r, sigma, q, option_type='call'):
        """Calculate all Greeks including first, second, and third order"""
        
        # Avoid division by zero
        if T <= 0:
            T = 0.0001
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # First-order Greeks
        if option_type == 'call':
            delta = np.exp(-q*T) * norm.cdf(d1)
            theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) - 
                    r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        else:
            delta = -np.exp(-q*T) * norm.cdf(-d1)
            theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) + 
                    r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        
        gamma = norm.pdf(d1)*np.exp(-q*T) / (S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.exp(-q*T)*np.sqrt(T) / 100
        rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option_type == 'call' else -d2) / 100
        
        # Second-order Greeks
        vanna = -norm.pdf(d1)*d2*np.exp(-q*T) / (sigma)
        charm = np.exp(-q*T) * (q*norm.cdf(d1 if option_type == 'call' else -d1) - 
                               norm.pdf(d1)*(2*(r-q)*T - d2*sigma*np.sqrt(T))/(2*T*sigma*np.sqrt(T)))
        vomma = vega * d1 * d2 / sigma
        
        # Third-order Greeks
        speed = -gamma / S * (d1 / (sigma*np.sqrt(T)) + 1)
        zomma = gamma * (d1*d2 - 1) / sigma
        color = -2*q*np.exp(-q*T)*norm.cdf(d1 if option_type == 'call' else -d1) + \
                2*np.exp(-q*T)*norm.pdf(d1)*(2*(r-q)*T - d2*sigma*np.sqrt(T))/(2*T*sigma*np.sqrt(T)*S)
        
        # Lambda (leverage)
        option_price = OptionPricingModels.black_scholes(S, K, T, r, sigma, q, option_type)
        lambda_greek = delta * S / option_price if option_price != 0 else 0
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho,
            'Lambda': lambda_greek,
            'Vanna': vanna,
            'Charm': charm,
            'Vomma': vomma,
            'Speed': speed,
            'Zomma': zomma,
            'Color': color
        }

# Strategy Builder
class StrategyBuilder:
    
    @staticmethod
    def create_payoff_diagram(strategies, S_range, current_price):
        """Create interactive payoff diagram for option strategies"""
        fig = go.Figure()
        
        total_payoff = np.zeros_like(S_range)
        total_cost = 0
        
        for strategy in strategies:
            strike = strategy['strike']
            premium = strategy['premium']
            quantity = strategy['quantity']
            option_type = strategy['type']
            position = strategy['position']
            
            if position == 'buy':
                cost = premium * quantity
                total_cost += cost
            else:
                cost = -premium * quantity
                total_cost += cost
            
            if option_type == 'call':
                if position == 'buy':
                    payoff = quantity * np.maximum(S_range - strike, 0) - cost
                else:
                    payoff = -quantity * np.maximum(S_range - strike, 0) - cost
            else:  # put
                if position == 'buy':
                    payoff = quantity * np.maximum(strike - S_range, 0) - cost
                else:
                    payoff = -quantity * np.maximum(strike - S_range, 0) - cost
            
            total_payoff += payoff
            
            # Add individual leg to plot (dotted lines)
            fig.add_trace(go.Scatter(
                x=S_range,
                y=payoff,
                mode='lines',
                name=f"{position.title()} {quantity} {option_type}(s) @ ${strike:.2f}",
                line=dict(dash='dot', width=1.5),
                opacity=0.4,
                visible='legendonly'
            ))
        
        # Add profit area (green)
        profit_indices = total_payoff > 0
        if np.any(profit_indices):
            fig.add_trace(go.Scatter(
                x=S_range[profit_indices],
                y=total_payoff[profit_indices],
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(16, 185, 129, 0.3)',
                name='Profit Area',
                showlegend=False
            ))
        
        # Add loss area (red)
        loss_indices = total_payoff <= 0
        if np.any(loss_indices):
            fig.add_trace(go.Scatter(
                x=S_range[loss_indices],
                y=total_payoff[loss_indices],
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(239, 68, 68, 0.3)',
                name='Loss Area',
                showlegend=False
            ))
        
        # Add total payoff line (thick black line)
        fig.add_trace(go.Scatter(
            x=S_range,
            y=total_payoff,
            mode='lines',
            name='Strategy Payoff',
            line=dict(color='#1e3c72', width=3)
        ))
        
        # Add break-even line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
        
        # Add current price line
        fig.add_vline(x=current_price, line_dash="dash", line_color="#2a5298", opacity=0.7,
                     annotation_text=f"Current: ${current_price:.2f}", annotation_position="top")
        
        # Calculate key metrics
        breakeven_points = []
        for i in range(1, len(S_range)):
            if total_payoff[i-1] * total_payoff[i] < 0:
                # Linear interpolation for more accurate breakeven
                x1, x2 = S_range[i-1], S_range[i]
                y1, y2 = total_payoff[i-1], total_payoff[i]
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(breakeven)
        
        # Add breakeven points
        for be in breakeven_points:
            fig.add_vline(x=be, line_dash="dot", line_color="orange", opacity=0.5,
                         annotation_text=f"B/E: ${be:.2f}", annotation_position="bottom")
        
        max_profit = np.max(total_payoff)
        max_loss = np.min(total_payoff)
        
        # Find optimal points
        max_profit_price = S_range[np.argmax(total_payoff)]
        max_loss_price = S_range[np.argmin(total_payoff)]
        
        # Add annotations for max profit/loss
        if max_profit != np.inf:
            fig.add_annotation(
                x=max_profit_price,
                y=max_profit,
                text=f"Max Profit: ${max_profit:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor="#10b981",
                ax=40,
                ay=-40,
                bgcolor="rgba(16, 185, 129, 0.8)",
                bordercolor="#10b981",
                font=dict(color="white", size=12)
            )
        
        if max_loss != -np.inf:
            fig.add_annotation(
                x=max_loss_price,
                y=max_loss,
                text=f"Max Loss: ${max_loss:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor="#ef4444",
                ax=-40,
                ay=40,
                bgcolor="rgba(239, 68, 68, 0.8)",
                bordercolor="#ef4444",
                font=dict(color="white", size=12)
            )
        
        fig.update_layout(
            title={
                'text': 'Option Strategy Payoff Diagram',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Stock Price ($)',
            yaxis_title='Profit/Loss ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor="rgba(255, 255, 255, 0.3)",
                borderwidth=1,
                font=dict(color="white")
            ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(family="Arial, sans-serif", color="white"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#333333',
                title_font=dict(family="Arial, sans-serif")
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#333333',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                title_font=dict(family="Arial, sans-serif")
            )
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
        
        return fig, {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_points': breakeven_points,
            'total_cost': total_cost,
            'max_profit_price': max_profit_price,
            'max_loss_price': max_loss_price
        }
    
    @staticmethod
    def get_predefined_strategies():
        """Get predefined option strategies"""
        return {
            'Long Call': [{'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 0}],
            'Long Put': [{'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': 0}],
            'Short Call': [{'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 0}],
            'Short Put': [{'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': 0}],
            'Covered Call': [{'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 5}],
            'Protective Put': [{'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -5}],
            'Bull Call Spread': [
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 0},
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 10}
            ],
            'Bear Put Spread': [
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': 0},
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': -10}
            ],
            'Bull Put Spread': [
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': 0},
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -10}
            ],
            'Bear Call Spread': [
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 0},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 10}
            ],
            'Long Straddle': [
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 0},
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': 0}
            ],
            'Short Straddle': [
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 0},
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': 0}
            ],
            'Long Strangle': [
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 10},
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -10}
            ],
            'Short Strangle': [
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 10},
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': -10}
            ],
            'Iron Condor': [
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': -10},
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -20},
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 10},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 20}
            ],
            'Iron Butterfly': [
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': 0},
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -10},
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 0},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 10}
            ],
            'Long Butterfly': [
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': -10},
                {'type': 'call', 'position': 'sell', 'quantity': 2, 'strike_offset': 0},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 10}
            ],
            'Calendar Spread': [
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 0, 'time_offset': -0.083},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 0, 'time_offset': 0}
            ],
            'Diagonal Spread': [
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 5, 'time_offset': -0.083},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 0, 'time_offset': 0}
            ],
            'Ratio Call Spread': [
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 0},
                {'type': 'call', 'position': 'sell', 'quantity': 2, 'strike_offset': 10}
            ],
            'Ratio Put Spread': [
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': 0},
                {'type': 'put', 'position': 'sell', 'quantity': 2, 'strike_offset': -10}
            ],
            'Jade Lizard': [
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 10},
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': -5},
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -15}
            ],
            'Reverse Iron Condor': [
                {'type': 'put', 'position': 'buy', 'quantity': 1, 'strike_offset': -10},
                {'type': 'put', 'position': 'sell', 'quantity': 1, 'strike_offset': -20},
                {'type': 'call', 'position': 'buy', 'quantity': 1, 'strike_offset': 10},
                {'type': 'call', 'position': 'sell', 'quantity': 1, 'strike_offset': 20}
            ]
        }

# Main Application
def main():
    # Create tabs with icons
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Option Pricing", "Greeks Analysis", "Strategy Builder", 
        "Monte Carlo", "Volatility Analysis", "Option Chain"
    ])
    
    # Tab 1: Option Pricing
    with tab1:
        st.markdown('<div class="info-card"><h3>Option Pricing Models</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            exercise_type = st.selectbox("Exercise Type", ["European", "American"])
        
        with col2:
            st.write("")  # Spacer
            calculate_btn = st.button("Calculate Prices", type="primary", use_container_width=True)
        
        if calculate_btn:
            with st.spinner("Calculating option prices..."):
                # Calculate prices using all models
                results = {}
                
                # Black-Scholes
                bs_price = OptionPricingModels.black_scholes(
                    spot_price, strike_price, time_to_maturity, risk_free_rate, 
                    volatility, dividend_yield, option_type.lower()
                )
                results['Black-Scholes'] = bs_price
                
                # Binomial Tree
                bt_price = OptionPricingModels.binomial_tree(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, binomial_steps, option_type.lower(),
                    american=(exercise_type == "American")
                )
                results['Binomial Tree'] = bt_price
                
                # Monte Carlo
                mc_price, mc_error = OptionPricingModels.monte_carlo(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, mc_simulations, option_type.lower()
                )
                results['Monte Carlo'] = mc_price
                
                # Heston Model
                heston_price, heston_error, _ = OptionPricingModels.heston_monte_carlo(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    dividend_yield, v0, kappa, theta, sigma_v, rho_heston,
                    mc_simulations, 100, option_type.lower()
                )
                results['Heston'] = heston_price
                
                # Merton Jump Diffusion
                merton_price = OptionPricingModels.merton_jump_diffusion(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, lambda_jump, mu_jump, sigma_jump,
                    50, option_type.lower()
                )
                results['Merton Jump'] = merton_price
                
                # Display results
                st.markdown('<div class="info-card"><h4>Pricing Results</h4></div>', unsafe_allow_html=True)
                
                # Create comparison chart
                fig = go.Figure()
                models = list(results.keys())
                prices = list(results.values())
                
                colors = ['#1e3c72', '#2a5298', '#3a6ab1', '#4a7aca', '#5a8ae3']
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=prices,
                    text=[f"${p:.4f}" for p in prices],
                    textposition='auto',
                    marker_color=colors,
                    marker_line_color='rgb(0,0,0)',
                    marker_line_width=1.5
                ))
                
                fig.update_layout(
                    title={
                        'text': "Option Price Comparison Across Models",
                        'font': {'size': 20, 'family': 'Arial, sans-serif'}
                    },
                    xaxis_title="Model",
                    yaxis_title="Option Price ($)",
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(family="Arial, sans-serif", color="white"),
                    xaxis=dict(gridcolor='#333333'),
                    yaxis=dict(gridcolor='#333333')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed results
                col1, col2, col3, col4, col5 = st.columns(5)
                cols = [col1, col2, col3, col4, col5]
                
                for i, (model, price) in enumerate(results.items()):
                    with cols[i]:
                        st.metric(model, f"${price:.4f}")
                
                # Additional metrics
                st.markdown('<div class="info-card"><h4>Additional Metrics</h4></div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    intrinsic_value = max(0, spot_price - strike_price) if option_type == "Call" else max(0, strike_price - spot_price)
                    st.metric("Intrinsic Value", f"${intrinsic_value:.2f}")
                
                with col2:
                    time_value = bs_price - intrinsic_value
                    st.metric("Time Value", f"${time_value:.2f}")
                
                with col3:
                    moneyness = spot_price / strike_price
                    st.metric("Moneyness", f"{moneyness:.3f}")
                
                with col4:
                    if spot_price > strike_price * 1.02:
                        status = "ITM" if option_type == "Call" else "OTM"
                    elif spot_price < strike_price * 0.98:
                        status = "OTM" if option_type == "Call" else "ITM"
                    else:
                        status = "ATM"
                    st.metric("Status", status)
    
    # Tab 2: Greeks Analysis
    with tab2:
        st.markdown('<div class="info-card"><h3>Greeks Analysis</h3></div>', unsafe_allow_html=True)
        
        # Calculate Greeks
        greeks = Greeks.calculate_all_greeks(
            spot_price, strike_price, time_to_maturity, risk_free_rate,
            volatility, dividend_yield, option_type.lower()
        )
        
        # Display Greeks in organized sections
        st.markdown('<div class="info-card"><h4>First-Order Greeks</h4></div>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Delta (Δ)", f"{greeks['Delta']:.4f}", 
                     help="Rate of change of option price with respect to underlying price")
        with col2:
            st.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}",
                     help="Rate of change of delta with respect to underlying price")
        with col3:
            st.metric("Theta (Θ)", f"{greeks['Theta']:.4f}",
                     help="Rate of change of option price with respect to time (per day)")
        with col4:
            st.metric("Vega (ν)", f"{greeks['Vega']:.4f}",
                     help="Rate of change of option price with respect to volatility")
        with col5:
            st.metric("Rho (ρ)", f"{greeks['Rho']:.4f}",
                     help="Rate of change of option price with respect to interest rate")
        with col6:
            st.metric("Lambda (λ)", f"{greeks['Lambda']:.4f}",
                     help="Percentage change in option price for 1% change in underlying")
        
        st.markdown('<div class="info-card"><h4>Second-Order Greeks</h4></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vanna", f"{greeks['Vanna']:.4f}",
                     help="Rate of change of delta with respect to volatility")
        with col2:
            st.metric("Charm", f"{greeks['Charm']:.4f}",
                     help="Rate of change of delta with respect to time")
        with col3:
            st.metric("Vomma", f"{greeks['Vomma']:.4f}",
                     help="Rate of change of vega with respect to volatility")
        
        st.markdown('<div class="info-card"><h4>Third-Order Greeks</h4></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Speed", f"{greeks['Speed']:.6f}",
                     help="Rate of change of gamma with respect to underlying price")
        with col2:
            st.metric("Zomma", f"{greeks['Zomma']:.4f}",
                     help="Rate of change of gamma with respect to volatility")
        with col3:
            st.metric("Color", f"{greeks['Color']:.6f}",
                     help="Rate of change of gamma with respect to time")
        
        # Greeks visualization
        st.markdown('<div class="info-card"><h4>Greeks Sensitivity Analysis</h4></div>', unsafe_allow_html=True)
        
        # Create interactive plots for Greeks
        spot_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 100)
        
        # Calculate Greeks across spot prices
        delta_values = []
        gamma_values = []
        theta_values = []
        vega_values = []
        
        for s in spot_range:
            g = Greeks.calculate_all_greeks(s, strike_price, time_to_maturity, risk_free_rate,
                                           volatility, dividend_yield, option_type.lower())
            delta_values.append(g['Delta'])
            gamma_values.append(g['Gamma'])
            theta_values.append(g['Theta'])
            vega_values.append(g['Vega'])
        
        # Create subplots for Greeks
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'),
                           vertical_spacing=0.3,
                           horizontal_spacing=0.1)
        
        # Delta plot
        fig.add_trace(go.Scatter(x=spot_range, y=delta_values, name='Delta',
                                line=dict(color="#06a800", width=3)), row=1, col=1)
        
        # Gamma plot
        fig.add_trace(go.Scatter(x=spot_range, y=gamma_values, name='Gamma',
                                line=dict(color="#ca2626", width=3)), row=1, col=2)
        
        # Theta plot
        fig.add_trace(go.Scatter(x=spot_range, y=theta_values, name='Theta',
                                line=dict(color="#0064fa", width=3)), row=2, col=1)
        
        # Vega plot
        fig.add_trace(go.Scatter(x=spot_range, y=vega_values, name='Vega',
                                line=dict(color="#fcfcfc", width=3)), row=2, col=2)
        
        # Add current spot price line
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=spot_price, line_dash="dash", line_color="gray",
                            opacity=0.5, row=row, col=col)
        
        fig.update_layout(height=600, showlegend=False, template='plotly_dark',
                         title={
                             'text': "Greeks Sensitivity to Spot Price",
                             'font': {'size': 20, 'family': 'Arial, sans-serif'}
                         },
                         plot_bgcolor='black',
                         paper_bgcolor='black',
                         font=dict(family="Arial, sans-serif", color="white"))
        fig.update_xaxes(title_text="Spot Price ($)", title_font=dict(family="Arial, sans-serif"),
                        gridcolor='#333333')
        fig.update_yaxes(gridcolor='#333333')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Greeks Surface
        st.markdown('<div class="info-card"><h4>3D Greeks Surface</h4></div>', unsafe_allow_html=True)
        
        # Create meshgrid for 3D plot
        spot_3d = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
        vol_3d = np.linspace(volatility * 0.5, volatility * 1.5, 50)
        X, Y = np.meshgrid(spot_3d, vol_3d)
        
        # Calculate delta surface
        Z = np.zeros_like(X)
        for i in range(len(spot_3d)):
            for j in range(len(vol_3d)):
                g = Greeks.calculate_all_greeks(X[j, i], strike_price, time_to_maturity,
                                               risk_free_rate, Y[j, i], dividend_yield,
                                               option_type.lower())
                Z[j, i] = g['Delta']
        
        fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
                                          colorscale='Jet',
                                          name='Delta Surface')])
        
        fig_3d.update_layout(
            title={
                'text': 'Delta Surface',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            scene=dict(
                xaxis_title='Spot Price ($)',
                yaxis_title='Volatility',
                zaxis_title='Delta',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis=dict(
                    backgroundcolor="black",
                    gridcolor="#333333",
                    showbackground=True,
                    zerolinecolor="#333333",
                    title_font=dict(family="Arial, sans-serif")
                ),
                yaxis=dict(
                    backgroundcolor="black",
                    gridcolor="#333333",
                    showbackground=True,
                    zerolinecolor="#333333",
                    title_font=dict(family="Arial, sans-serif")
                ),
                zaxis=dict(
                    backgroundcolor="black",
                    gridcolor="#333333",
                    showbackground=True,
                    zerolinecolor="#333333",
                    title_font=dict(family="Arial, sans-serif")
                ),
                bgcolor="black"
            ),
            height=600,
            paper_bgcolor='black',
            font=dict(family="Arial, sans-serif", color="white")
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Tab 3: Strategy Builder
    with tab3:
        st.markdown('<div class="info-card"><h3>Option Strategy Builder</h3></div>', unsafe_allow_html=True)
        
        # Strategy selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            strategy_type = st.selectbox(
                "Select Strategy Type",
                ["Predefined Strategies", "Custom Strategy"]
            )
        
        strategies_list = []
        
        if strategy_type == "Predefined Strategies":
            predefined = StrategyBuilder.get_predefined_strategies()
            
            with col2:
                selected_strategy = st.selectbox("Choose Strategy", list(predefined.keys()))
            
            st.markdown(f'<div class="info-card"><h4>Building {selected_strategy}</h4></div>', unsafe_allow_html=True)
            
            # Build strategy legs
            for i, leg in enumerate(predefined[selected_strategy]):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    strike = spot_price + leg['strike_offset']
                    st.info(f"**Leg {i+1} Strike:** ${strike:.2f}")
                
                with col2:
                    st.info(f"**Type:** {leg['type'].title()}")
                
                with col3:
                    st.info(f"**Position:** {leg['position'].title()}")
                
                with col4:
                    st.info(f"**Quantity:** {leg['quantity']}")
                
                # Calculate premium for this leg
                t_exp = time_to_maturity + leg.get('time_offset', 0)
                premium = OptionPricingModels.black_scholes(
                    spot_price, strike, t_exp, risk_free_rate,
                    volatility, dividend_yield, leg['type']
                )
                
                strategies_list.append({
                    'strike': strike,
                    'premium': premium,
                    'quantity': leg['quantity'],
                    'type': leg['type'],
                    'position': leg['position']
                })
        
        else:  # Custom Strategy
            st.markdown('<div class="info-card"><h4>Build Your Custom Strategy</h4></div>', unsafe_allow_html=True)
            
            # Option chain simulation
            strikes = np.arange(spot_price * 0.8, spot_price * 1.2, 2.5)
            
            # Create option chain
            option_chain = []
            for strike in strikes:
                call_price = OptionPricingModels.black_scholes(
                    spot_price, strike, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, 'call'
                )
                put_price = OptionPricingModels.black_scholes(
                    spot_price, strike, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, 'put'
                )
                
                call_greeks = Greeks.calculate_all_greeks(
                    spot_price, strike, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, 'call'
                )
                put_greeks = Greeks.calculate_all_greeks(
                    spot_price, strike, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, 'put'
                )
                
                option_chain.append({
                    'Strike': strike,
                    'Call Bid': call_price * 0.98,
                    'Call Ask': call_price * 1.02,
                    'Call IV': volatility * 100,
                    'Call Delta': call_greeks['Delta'],
                    'Put Bid': put_price * 0.98,
                    'Put Ask': put_price * 1.02,
                    'Put IV': volatility * 100,
                    'Put Delta': put_greeks['Delta']
                })
            
            df_chain = pd.DataFrame(option_chain)
            
            # Display option chain
            st.markdown('<div class="info-card"><h4>Option Chain</h4></div>', unsafe_allow_html=True)
            
            # Format the dataframe
            df_display = df_chain.copy()
            for col in ['Call Bid', 'Call Ask', 'Put Bid', 'Put Ask']:
                df_display[col] = df_display[col].apply(lambda x: f"${x:.2f}")
            for col in ['Call IV', 'Put IV']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")
            for col in ['Call Delta', 'Put Delta']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
            df_display['Strike'] = df_display['Strike'].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(df_display, use_container_width=True, height=300)
            
            # Custom strategy builder
            st.markdown('<div class="info-card"><h4>Add Legs to Your Strategy</h4></div>', unsafe_allow_html=True)
            
            num_legs = st.number_input("Number of legs", min_value=1, max_value=10, value=2)
            
            for i in range(num_legs):
                with st.container():
                    st.write(f"**Leg {i+1}**")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        leg_strike = st.selectbox(f"Strike", strikes,
                                                key=f"strike_{i}")
                    
                    with col2:
                        leg_type = st.selectbox(f"Type", ["call", "put"],
                                              key=f"type_{i}")
                    
                    with col3:
                        leg_position = st.selectbox(f"Position", ["buy", "sell"],
                                                  key=f"position_{i}")
                    
                    with col4:
                        leg_quantity = st.number_input(f"Quantity", min_value=1,
                                                     max_value=100, value=1,
                                                     key=f"quantity_{i}")
                    
                    with col5:
                        # Get premium from option chain
                        idx = np.argmin(np.abs(strikes - leg_strike))
                        if leg_type == "call":
                            premium = option_chain[idx]['Call Ask'] if leg_position == "buy" else option_chain[idx]['Call Bid']
                        else:
                            premium = option_chain[idx]['Put Ask'] if leg_position == "buy" else option_chain[idx]['Put Bid']
                        
                        st.metric("Premium", f"${premium:.2f}")
                    
                    strategies_list.append({
                        'strike': leg_strike,
                        'premium': premium,
                        'quantity': leg_quantity,
                        'type': leg_type,
                        'position': leg_position
                    })
        
        # Visualize strategy
        if st.button("🎯 Analyze Strategy", type="primary", use_container_width=True):
            if strategies_list:
                # Create payoff diagram
                S_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 500)
                
                fig, metrics = StrategyBuilder.create_payoff_diagram(
                    strategies_list, S_range, spot_price
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display strategy metrics
                st.markdown('<div class="info-card"><h4>Strategy Metrics</h4></div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Max Profit", f"${metrics['max_profit']:.2f}" if metrics['max_profit'] != np.inf else "Unlimited",
                             help=f"At stock price: ${metrics['max_profit_price']:.2f}" if metrics['max_profit'] != np.inf else "")
                
                with col2:
                    st.metric("Max Loss", f"${metrics['max_loss']:.2f}" if metrics['max_loss'] != -np.inf else "Unlimited",
                             help=f"At stock price: ${metrics['max_loss_price']:.2f}" if metrics['max_loss'] != -np.inf else "")
                
                with col3:
                    st.metric("Net Cost/Credit", f"${metrics['total_cost']:.2f}",
                             help="Positive = Net Debit, Negative = Net Credit")
                
                with col4:
                    be_points = ", ".join([f"${x:.2f}" for x in metrics['breakeven_points']])
                    st.metric("Breakeven", be_points if be_points else "None")
                
                # Probability analysis
                st.markdown('<div class="info-card"><h4>Probability Analysis</h4></div>', unsafe_allow_html=True)
                
                # Calculate probabilities
                days_to_exp = int(time_to_maturity * 365)
                
                # Probability of profit
                if len(metrics['breakeven_points']) > 0:
                    if len(metrics['breakeven_points']) == 1:
                        be = metrics['breakeven_points'][0]
                        if metrics['max_profit_price'] > be:
                            prob_profit = 1 - norm.cdf(
                                (np.log(be/spot_price) + (risk_free_rate - 0.5*volatility**2)*time_to_maturity) /
                                (volatility*np.sqrt(time_to_maturity))
                            )
                        else:
                            prob_profit = norm.cdf(
                                (np.log(be/spot_price) + (risk_free_rate - 0.5*volatility**2)*time_to_maturity) /
                                (volatility*np.sqrt(time_to_maturity))
                            )
                    else:
                        # Multiple breakeven points
                        be1, be2 = metrics['breakeven_points'][0], metrics['breakeven_points'][1]
                        prob_profit = norm.cdf(
                            (np.log(be2/spot_price) + (risk_free_rate - 0.5*volatility**2)*time_to_maturity) /
                            (volatility*np.sqrt(time_to_maturity))
                        ) - norm.cdf(
                            (np.log(be1/spot_price) + (risk_free_rate - 0.5*volatility**2)*time_to_maturity) /
                            (volatility*np.sqrt(time_to_maturity))
                        )
                else:
                    prob_profit = 0.5  # Default if no breakeven
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Probability of Profit", f"{prob_profit*100:.1f}%",
                             help="Based on log-normal distribution")
                
                with col2:
                    st.metric("Days to Expiration", days_to_exp)
                
                with col3:
                    expected_move = spot_price * volatility * np.sqrt(time_to_maturity)
                    st.metric("Expected Move", f"±${expected_move:.2f}",
                             help="1 standard deviation move")
    
    # Tab 4: Monte Carlo Simulation
    with tab4:
        st.markdown('<div class="info-card"><h3>Monte Carlo Simulation</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            mc_option_type = st.selectbox("Option Type for MC", ["Call", "Put"])
            show_paths = st.checkbox("Show Sample Paths", value=True)
        
        with col2:
            num_paths_show = st.slider("Number of Paths to Display", 10, 100, 50)
        
        if st.button("Run Monte Carlo Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                # Standard Monte Carlo
                mc_price, mc_error = OptionPricingModels.monte_carlo(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, mc_simulations, mc_option_type.lower()
                )
                
                # Heston Monte Carlo with paths
                heston_price, heston_error, paths = OptionPricingModels.heston_monte_carlo(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    dividend_yield, v0, kappa, theta, sigma_v, rho_heston,
                    mc_simulations, 100, mc_option_type.lower()
                )
                
                # Display results
                st.markdown('<div class="info-card"><h4>Simulation Results</h4></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("GBM Monte Carlo Price", f"${mc_price:.4f}")
                    st.metric("Standard Error", f"${mc_error:.4f}")
                    confidence_interval = 1.96 * mc_error
                    st.metric("95% Confidence Interval", f"±${confidence_interval:.4f}")
                
                with col2:
                    st.metric("Heston Monte Carlo Price", f"${heston_price:.4f}")
                    st.metric("Standard Error", f"${heston_error:.4f}")
                    heston_confidence = 1.96 * heston_error
                    st.metric("95% Confidence Interval", f"±${heston_confidence:.4f}")
                
                # Visualize paths
                if show_paths:
                    st.markdown('<div class="info-card"><h4>Sample Price Paths</h4></div>', unsafe_allow_html=True)
                    
                    # Create time array
                    time_steps = np.linspace(0, time_to_maturity, 101)
                    
                    # Plot sample paths
                    fig = go.Figure()
                    
                    # Add sample paths
                    for i in range(min(num_paths_show, len(paths))):
                        fig.add_trace(go.Scatter(
                            x=time_steps,
                            y=paths[i],
                            mode='lines',
                            line=dict(width=0.5),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Add mean path
                    mean_path = np.mean(paths[:num_paths_show], axis=0)
                    fig.add_trace(go.Scatter(
                        x=time_steps,
                        y=mean_path,
                        mode='lines',
                        line=dict(color='#e74c3c', width=3),
                        name='Mean Path'
                    ))
                    
                    # Add strike price line
                    fig.add_hline(y=strike_price, line_dash="dash", line_color="#9aff60",
                                 annotation_text=f"Strike: ${strike_price}")
                    
                    # Add current price line
                    fig.add_hline(y=spot_price, line_dash="dash", line_color="#0099ff",
                                 annotation_text=f"Current: ${spot_price}")
                    
                    fig.update_layout(
                        title={
                            'text': "Simulated Stock Price Paths (Heston Model)",
                            'font': {'size': 20, 'family': 'Arial, sans-serif'}
                        },
                        xaxis_title="Time (Years)",
                        yaxis_title="Stock Price ($)",
                        template='plotly_dark',
                        height=500,
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        font=dict(family="Arial, sans-serif", color="white"),
                        xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
                        yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Terminal price distribution
                st.markdown('<div class="info-card"><h4>Terminal Price Distribution</h4></div>', unsafe_allow_html=True)
                
                # Generate terminal prices for histogram
                np.random.seed(42)
                Z = np.random.standard_normal(10000)
                ST_gbm = spot_price * np.exp((risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_maturity + 
                                            volatility * np.sqrt(time_to_maturity) * Z)
                
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=ST_gbm,
                    nbinsx=50,
                    name='GBM Distribution',
                    marker_color='#3498db',
                    opacity=0.7
                ))
                
                fig_hist.add_vline(x=strike_price, line_dash="dash", line_color="#e74c3c",
                                  annotation_text=f"Strike: ${strike_price}")
                fig_hist.add_vline(x=spot_price, line_dash="dash", line_color="#2ecc71",
                                  annotation_text=f"Current: ${spot_price}")
                
                fig_hist.update_layout(
                    title={
                        'text': "Terminal Price Distribution",
                        'font': {'size': 20, 'family': 'Arial, sans-serif'}
                    },
                    xaxis_title="Stock Price at Maturity ($)",
                    yaxis_title="Frequency",
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(family="Arial, sans-serif", color="white"),
                    xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
                    yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Convergence analysis
                st.markdown('<div class="info-card"><h4>Monte Carlo Convergence Analysis</h4></div>', unsafe_allow_html=True)
                
                convergence_points = [100, 500, 1000, 2500, 5000, 10000, 25000, 50000]
                convergence_prices = []
                convergence_errors = []
                
                progress_bar = st.progress(0)
                for idx, n in enumerate(convergence_points):
                    if n <= mc_simulations:
                        price, error = OptionPricingModels.monte_carlo(
                            spot_price, strike_price, time_to_maturity, risk_free_rate,
                            volatility, dividend_yield, n, mc_option_type.lower()
                        )
                        convergence_prices.append(price)
                        convergence_errors.append(error)
                        progress_bar.progress((idx + 1) / len(convergence_points))
                
                progress_bar.empty()
                
                fig_conv = go.Figure()
                
                fig_conv.add_trace(go.Scatter(
                    x=convergence_points[:len(convergence_prices)],
                    y=convergence_prices,
                    mode='lines+markers',
                    name='MC Price',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=8),
                    error_y=dict(
                        type='data',
                        array=[1.96*e for e in convergence_errors],
                        visible=True,
                        color='rgba(52, 152, 219, 0.3)'
                    )
                ))
                
                # Add Black-Scholes price as reference
                bs_ref = OptionPricingModels.black_scholes(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    volatility, dividend_yield, mc_option_type.lower()
                )
                fig_conv.add_hline(y=bs_ref, line_dash="dash", line_color="#e74c3c",
                                  annotation_text=f"Black-Scholes: ${bs_ref:.4f}")
                
                fig_conv.update_layout(
                    title={
                        'text': "Monte Carlo Convergence",
                        'font': {'size': 20, 'family': 'Arial, sans-serif'}
                    },
                    xaxis_title="Number of Simulations",
                    yaxis_title="Option Price ($)",
                    xaxis_type="log",
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(family="Arial, sans-serif", color="white"),
                    xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
                    yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
                )
                
                st.plotly_chart(fig_conv, use_container_width=True)
    
    # Tab 5: Volatility Analysis
    with tab5:
        st.markdown('<div class="info-card"><h3>Volatility Analysis</h3></div>', unsafe_allow_html=True)
        
        # Implied Volatility Calculator
        st.markdown('<div class="info-card"><h4>Implied Volatility Calculator</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_price = st.number_input("Market Price ($)", value=5.0, min_value=0.01, step=0.01)
        
        with col2:
            iv_option_type = st.selectbox("Option Type for IV", ["Call", "Put"])
        
        with col3:
            calculate_iv = st.button("Calculate IV", type="primary")
        
        if calculate_iv:
            # Newton-Raphson method for IV
            def objective(vol):
                return OptionPricingModels.black_scholes(
                    spot_price, strike_price, time_to_maturity, risk_free_rate,
                    vol, dividend_yield, iv_option_type.lower()
                ) - market_price
            
            try:
                result = minimize_scalar(lambda x: abs(objective(x)), bounds=(0.001, 5), method='bounded')
                implied_vol = result.x
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"Implied Volatility: {implied_vol*100:.2f}%")
                with col2:
                    calculated_price = OptionPricingModels.black_scholes(
                        spot_price, strike_price, time_to_maturity, risk_free_rate,
                        implied_vol, dividend_yield, iv_option_type.lower()
                    )
                    st.info(f"Calculated Price: ${calculated_price:.4f}")
                with col3:
                    error = abs(calculated_price - market_price)
                    st.info(f"Pricing Error: ${error:.6f}")
            except:
                st.error("Could not calculate implied volatility. Check if market price is reasonable.")
        
        # Volatility Smile
        st.markdown('<div class="info-card"><h4>Volatility Smile/Skew</h4></div>', unsafe_allow_html=True)
        
        # Generate volatility smile
        strikes_smile = np.linspace(spot_price * 0.7, spot_price * 1.3, 30)
        implied_vols = []
        
        for k in strikes_smile:
            # Simulate market prices with volatility smile
            moneyness = k / spot_price
            if moneyness < 1:
                vol_smile = volatility * (1 + 0.2 * (1 - moneyness))
            else:
                vol_smile = volatility * (1 + 0.1 * (moneyness - 1))
            
            implied_vols.append(vol_smile)
        
        fig_smile = go.Figure()
        
        fig_smile.add_trace(go.Scatter(
            x=strikes_smile,
            y=[iv * 100 for iv in implied_vols],
            mode='lines+markers',
            name='Implied Volatility',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=6)
        ))
        
        fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="#e74c3c",
                           annotation_text=f"Spot: ${spot_price}")
        
        fig_smile.update_layout(
            title={
                'text': "Implied Volatility Smile",
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility (%)",
            template='plotly_dark',
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(family="Arial, sans-serif", color="white"),
            xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
            yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
        )
        
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # Term Structure
        st.markdown('<div class="info-card"><h4>Volatility Term Structure</h4></div>', unsafe_allow_html=True)
        
        maturities = [7, 14, 30, 60, 90, 120, 180, 365]
        term_vols = []
        
        for days in maturities:
            # Simulate term structure
            if days <= 30:
                vol_term = volatility * 1.2
            elif days <= 90:
                vol_term = volatility * 1.1
            else:
                vol_term = volatility * (1 + 0.05 * np.log(days/90))
            
            term_vols.append(vol_term)
        
        fig_term = go.Figure()
        
        fig_term.add_trace(go.Scatter(
            x=maturities,
            y=[tv * 100 for tv in term_vols],
            mode='lines+markers',
            name='Term Structure',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=8)
        ))
        
        fig_term.update_layout(
            title={
                'text': "Volatility Term Structure",
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Days to Maturity",
            yaxis_title="Implied Volatility (%)",
            template='plotly_dark',
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(family="Arial, sans-serif", color="white"),
            xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
            yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
        )
        
        st.plotly_chart(fig_term, use_container_width=True)
        
        # Volatility Surface
        st.markdown('<div class="info-card"><h4>3D Volatility Surface</h4></div>', unsafe_allow_html=True)
        
        # Create meshgrid for surface
        strikes_surf = np.linspace(spot_price * 0.8, spot_price * 1.2, 20)
        maturities_surf = np.linspace(0.1, 1, 20)
        X_surf, Y_surf = np.meshgrid(strikes_surf, maturities_surf)
        
        # Generate volatility surface
        Z_surf = np.zeros_like(X_surf)
        for i in range(len(strikes_surf)):
            for j in range(len(maturities_surf)):
                moneyness = X_surf[j, i] / spot_price
                time_factor = np.sqrt(Y_surf[j, i])
                
                if moneyness < 1:
                    Z_surf[j, i] = volatility * (1 + 0.2 * (1 - moneyness) * time_factor)
                else:
                    Z_surf[j, i] = volatility * (1 + 0.1 * (moneyness - 1) * time_factor)
        
        fig_surf = go.Figure(data=[go.Surface(
            x=X_surf,
            y=Y_surf,
            z=Z_surf * 100,
            colorscale='Jet',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
            )
        )])
        
        fig_surf.update_layout(
            title={
                'text': 'Implied Volatility Surface',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Time to Maturity (Years)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis=dict(
                    backgroundcolor="black",
                    gridcolor="#333333",
                    showbackground=True,
                    zerolinecolor="#333333",
                    title_font=dict(family="Arial, sans-serif")
                ),
                yaxis=dict(
                    backgroundcolor="black",
                    gridcolor="#333333",
                    showbackground=True,
                    zerolinecolor="#333333",
                    title_font=dict(family="Arial, sans-serif")
                ),
                zaxis=dict(
                    backgroundcolor="black",
                    gridcolor="#333333",
                    showbackground=True,
                    zerolinecolor="#333333",
                    title_font=dict(family="Arial, sans-serif")
                ),
                bgcolor="black"
            ),
            height=600,
            paper_bgcolor='black',
            font=dict(family="Arial, sans-serif", color="white")
        )
        
        st.plotly_chart(fig_surf, use_container_width=True)
    
    # Tab 6: Option Chain
    with tab6:
        st.markdown('<div class="info-card"><h3>Interactive Option Chain</h3></div>', unsafe_allow_html=True)
        
        # Generate comprehensive option chain
        chain_strikes = np.arange(spot_price * 0.7, spot_price * 1.3, 1)
        
        option_chain_data = []
        
        for strike in chain_strikes:
            # Calculate prices and Greeks for calls and puts
            call_price = OptionPricingModels.black_scholes(
                spot_price, strike, time_to_maturity, risk_free_rate,
                volatility, dividend_yield, 'call'
            )
            put_price = OptionPricingModels.black_scholes(
                spot_price, strike, time_to_maturity, risk_free_rate,
                volatility, dividend_yield, 'put'
            )
            
            call_greeks = Greeks.calculate_all_greeks(
                spot_price, strike, time_to_maturity, risk_free_rate,
                volatility, dividend_yield, 'call'
            )
            put_greeks = Greeks.calculate_all_greeks(
                spot_price, strike, time_to_maturity, risk_free_rate,
                volatility, dividend_yield, 'put'
            )
            
            # Simulate bid-ask spreads
            call_bid = call_price * 0.98
            call_ask = call_price * 1.02
            put_bid = put_price * 0.98
            put_ask = put_price * 1.02
            
            # Simulate volume and open interest
            moneyness = strike / spot_price
            volume_factor = np.exp(-2 * (moneyness - 1)**2)
            call_volume = int(10000 * volume_factor * np.random.uniform(0.8, 1.2))
            put_volume = int(8000 * volume_factor * np.random.uniform(0.8, 1.2))
            
            option_chain_data.append({
                'Strike': strike,
                'Call Bid': call_bid,
                'Call Ask': call_ask,
                'Call Mid': (call_bid + call_ask) / 2,
                'Call Volume': call_volume,
                'Call OI': call_volume * 10,
                'Call IV': volatility * 100,
                'Call Delta': call_greeks['Delta'],
                'Call Gamma': call_greeks['Gamma'],
                'Call Theta': call_greeks['Theta'],
                'Call Vega': call_greeks['Vega'],
                'Put Bid': put_bid,
                'Put Ask': put_ask,
                'Put Mid': (put_bid + put_ask) / 2,
                'Put Volume': put_volume,
                'Put OI': put_volume * 10,
                'Put IV': volatility * 100,
                'Put Delta': put_greeks['Delta'],
                'Put Gamma': put_greeks['Gamma'],
                'Put Theta': put_greeks['Theta'],
                'Put Vega': put_greeks['Vega']
            })
        
        df_full_chain = pd.DataFrame(option_chain_data)
        
        # Filter options
        st.markdown('<div class="info-card"><h4>Filter Options</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            moneyness_filter = st.selectbox(
                "Moneyness Filter",
                ["All", "ITM", "ATM", "OTM"]
            )
        
        with col2:
            min_volume = st.number_input("Min Volume", value=0, min_value=0)
        
        with col3:
            show_greeks = st.checkbox("Show Greeks", value=True)
        
        # Apply filters
        filtered_chain = df_full_chain.copy()
        
        if moneyness_filter == "ITM":
            filtered_chain = filtered_chain[
                ((filtered_chain['Strike'] < spot_price) & (option_type == "Call")) |
                ((filtered_chain['Strike'] > spot_price) & (option_type == "Put"))
            ]
        elif moneyness_filter == "ATM":
            filtered_chain = filtered_chain[
                np.abs(filtered_chain['Strike'] - spot_price) <= spot_price * 0.05
            ]
        elif moneyness_filter == "OTM":
            filtered_chain = filtered_chain[
                ((filtered_chain['Strike'] > spot_price) & (option_type == "Call")) |
                ((filtered_chain['Strike'] < spot_price) & (option_type == "Put"))
            ]
        
        filtered_chain = filtered_chain[
            (filtered_chain['Call Volume'] >= min_volume) |
            (filtered_chain['Put Volume'] >= min_volume)
        ]
        
        # Display option chain
        st.markdown('<div class="info-card"><h4>Option Chain Data</h4></div>', unsafe_allow_html=True)
        
        # Select columns to display
        if show_greeks:
            display_columns = ['Strike', 'Call Bid', 'Call Ask', 'Call Volume', 'Call IV',
                             'Call Delta', 'Call Gamma', 'Call Theta', 'Call Vega',
                             'Put Bid', 'Put Ask', 'Put Volume', 'Put IV',
                             'Put Delta', 'Put Gamma', 'Put Theta', 'Put Vega']
        else:
            display_columns = ['Strike', 'Call Bid', 'Call Ask', 'Call Volume', 'Call IV',
                             'Put Bid', 'Put Ask', 'Put Volume', 'Put IV']
        
        # Format the display
        df_display = filtered_chain[display_columns].copy()
        
        # Format currency columns
        for col in ['Strike', 'Call Bid', 'Call Ask', 'Put Bid', 'Put Ask']:
            df_display[col] = df_display[col].apply(lambda x: f"${x:.2f}")
        
        # Format IV columns
        for col in ['Call IV', 'Put IV']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")
        
        # Format Greeks
        if show_greeks:
            for col in ['Call Delta', 'Call Gamma', 'Put Delta', 'Put Gamma']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
            for col in ['Call Theta', 'Put Theta', 'Call Vega', 'Put Vega']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        
        # Highlight ATM strikes
        def highlight_atm(row):
            strike_val = float(row['Strike'].replace('$' , ''))
            if abs(strike_val - spot_price) <= 2.5:
                return ['background-color: #e6f3ff'] * len(row)
            return [''] * len(row)
        
        styled_df = df_display.style.apply(highlight_atm, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Option Chain Visualizations
        st.markdown('<div class="info-card"><h4>Option Chain Analytics</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume distribution
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Bar(
                x=filtered_chain['Strike'],
                y=filtered_chain['Call Volume'],
                name='Call Volume',
                marker_color='#2ecc71',
                opacity=0.7
            ))
            
            fig_vol.add_trace(go.Bar(
                x=filtered_chain['Strike'],
                y=-filtered_chain['Put Volume'],
                name='Put Volume',
                marker_color='#e74c3c',
                opacity=0.7
            ))
            
            fig_vol.add_vline(x=spot_price, line_dash="dash", line_color="#3498db",
                            annotation_text=f"Spot: ${spot_price}")
            
            fig_vol.update_layout(
                title={
                    'text': "Volume Distribution",
                    'font': {'size': 20, 'family': 'Arial, sans-serif'}
                },
                xaxis_title="Strike Price ($)",
                yaxis_title="Volume",
                template='plotly_dark',
                height=400,
                barmode='overlay',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(family="Arial, sans-serif", color="white"),
                xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
                yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            # Open Interest distribution
            fig_oi = go.Figure()
            
            fig_oi.add_trace(go.Bar(
                x=filtered_chain['Strike'],
                y=filtered_chain['Call OI'],
                name='Call OI',
                marker_color='#27ae60',
                opacity=0.7
            ))
            
            fig_oi.add_trace(go.Bar(
                x=filtered_chain['Strike'],
                y=-filtered_chain['Put OI'],
                name='Put OI',
                marker_color='#c0392b',
                opacity=0.7
            ))
            
            fig_oi.add_vline(x=spot_price, line_dash="dash", line_color="#3498db",
                            annotation_text=f"Spot: ${spot_price}")
            
            fig_oi.update_layout(
                title={
                    'text': "Open Interest Distribution",
                    'font': {'size': 20, 'family': 'Arial, sans-serif'}
                },
                xaxis_title="Strike Price ($)",
                yaxis_title="Open Interest",
                template='plotly_dark',
                height=400,
                barmode='overlay',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(family="Arial, sans-serif", color="white"),
                xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
                yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
            )
            
            st.plotly_chart(fig_oi, use_container_width=True)
        
        # Max Pain Calculation
        st.markdown('<div class="info-card"><h4>Max Pain Analysis</h4></div>', unsafe_allow_html=True)
        
        max_pain_strikes = []
        max_pain_values = []
        
        for test_strike in chain_strikes:
            total_pain = 0
            
            for idx, row in df_full_chain.iterrows():
                strike = row['Strike']
                # Call pain
                if test_strike > strike:
                    call_pain = (test_strike - strike) * row['Call OI']
                    total_pain += call_pain
                
                # Put pain
                if test_strike < strike:
                    put_pain = (strike - test_strike) * row['Put OI']
                    total_pain += put_pain
            
            max_pain_strikes.append(test_strike)
            max_pain_values.append(total_pain)
        
        max_pain_strike = max_pain_strikes[np.argmin(max_pain_values)]
        
        fig_pain = go.Figure()
        
        fig_pain.add_trace(go.Scatter(
            x=max_pain_strikes,
            y=max_pain_values,
            mode='lines',
            name='Total Pain',
            line=dict(color='#8e44ad', width=3)
        ))
        
        fig_pain.add_vline(x=max_pain_strike, line_dash="dash", line_color='#e74c3c',
                          annotation_text=f"Max Pain: ${max_pain_strike:.2f}")
        fig_pain.add_vline(x=spot_price, line_dash="dash", line_color='#3498db',
                          annotation_text=f"Spot: ${spot_price}")
        
        fig_pain.update_layout(
            title={
                'text': "Max Pain Analysis",
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Strike Price ($)",
            yaxis_title="Total Pain Value",
            template='plotly_dark',
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(family="Arial, sans-serif", color="white"),
            xaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif")),
            yaxis=dict(gridcolor='#333333', title_font=dict(family="Arial, sans-serif"))
        )
        
        st.plotly_chart(fig_pain, use_container_width=True)
        
        # Put-Call Ratio
        total_call_volume = filtered_chain['Call Volume'].sum()
        total_put_volume = filtered_chain['Put Volume'].sum()
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        total_call_oi = filtered_chain['Call OI'].sum()
        total_put_oi = filtered_chain['Put OI'].sum()
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Put/Call Volume Ratio", f"{put_call_ratio:.2f}",
                     help="Ratio > 1 indicates bearish sentiment")
        
        with col2:
            st.metric("Put/Call OI Ratio", f"{put_call_oi_ratio:.2f}",
                     help="Ratio > 1 indicates bearish positioning")
        
        with col3:
            st.metric("Max Pain Strike", f"${max_pain_strike:.2f}",
                     help="Price where option writers suffer minimum loss")
        
        with col4:
            distance_to_max_pain = ((max_pain_strike - spot_price) / spot_price) * 100
            st.metric("Distance to Max Pain", f"{distance_to_max_pain:.1f}%",
                     help="Positive = Max pain above current price")

# Add footer with developer information
st.markdown("""
    <div class="developer-credit">
        <h3>Developed by Arnav Jain</h3>
        <p>Connect with me on <a href="https://www.linkedin.com/in/arnavj19/" target="_blank">LinkedIn</a></p>
        <p style="margin-top: 1rem; opacity: 0.8;">
            GammaX V1.0 | For educational purposes only. Not financial advice.
        </p>
    </div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
