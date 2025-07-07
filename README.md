# GammaX

**Advanced Option Pricing and Analysis Tool**

GammaX is a comprehensive derivatives analytics platform that provides institutional-quality option pricing, Greeks analysis, and strategy-building capabilities in an intuitive web interface. Designed for traders, analysts, and students, it combines sophisticated mathematical models with user-friendly visualizations. Built with Python and Streamlit.

---

## Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
  - [Multiple Pricing Models](#multiple-pricing-models)  
  - [Comprehensive Greeks Analysis](#comprehensive-greeks-analysis)  
  - [Strategy Builder](#strategy-builder)  
  - [Monte Carlo Simulation Engine](#monte-carlo-simulation-engine)  
  - [Volatility Analysis Suite](#volatility-analysis-suite)  
  - [Interactive Option Chain](#interactive-option-chain)  
- [Technical Implementation](#technical-implementation)  
  - [Technologies Used](#technologies-used)  
  - [Mathematical Models](#mathematical-models)  
  - [Design Features](#design-features)  
- [Installation & Usage](#installation--usage)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Running the Application](#running-the-application)  
  - [Usage Guide](#usage-guide)  
- [Contributing](#contributing)  
- [License](#license)  
- [Author](#author)  
- [Acknowledgments](#acknowledgments)  
- [Disclaimer](#disclaimer)  

---

## Overview

GammaX is a comprehensive derivatives analytics platform that provides institutional-quality option pricing, Greeks analysis, and strategy-building capabilities in an intuitive web interface. Designed for traders, analysts, and students, it combines sophisticated mathematical models with user-friendly visualizations. Built with Python and Streamlit.

---

## Key Features

### Multiple Pricing Models
- **Black-Scholes Model**: The industry-standard model for European options  
- **Binomial Tree Model**: Supports both European and American option pricing  
- **Monte Carlo Simulation**: Flexible simulation-based pricing with confidence intervals  
- **Heston Stochastic Volatility Model**: Advanced model accounting for volatility smile  
- **Merton Jump Diffusion Model**: Captures sudden price movements and jumps  

  ![2](https://github.com/user-attachments/assets/436ff768-e5ae-493f-aba9-74b33b64a7e1)

### Comprehensive Greeks Analysis
- **First-Order Greeks**: Delta, Gamma, Theta, Vega, Rho, Lambda  
- **Second-Order Greeks**: Vanna, Charm, Vomma  
- **Third-Order Greeks**: Speed, Zomma, Color  
- **Interactive Visualizations**: 2D sensitivity plots and 3D surface visualizations  
- **Real-time Calculations**: Greeks update dynamically with parameter changes

  ![3](https://github.com/user-attachments/assets/6107b2da-047b-4a01-bdeb-6981f021b008)
  ![4](https://github.com/user-attachments/assets/e6346c7b-4c25-43c7-b648-b57ffeb8ae53)
  ![5](https://github.com/user-attachments/assets/01698a2f-dc08-490e-9296-c9b0f07eb5f0)

### Strategy Builder
- **20+ Predefined Strategies**: Including spreads, straddles, strangles, butterflies, condors  
- **Custom Strategy Builder**: Create any combination of options positions  
- **Interactive Payoff Chart**: Visualize profit and loss zones  
- **Probability Analysis**: Calculate probability of profit based on log-normal distribution

  ![6](https://github.com/user-attachments/assets/6d50c2c7-1039-4671-88ea-788b184a4316)
  ![7](https://github.com/user-attachments/assets/7470dd7c-c144-44f9-939d-6f4a54507433)
  ![8](https://github.com/user-attachments/assets/6f853472-d2fa-473b-b209-14e519d6288a)

### Monte Carlo Simulation Engine
- **Path Visualization**: Display sample price paths with mean trajectory  
- **Convergence Analysis**: Visualize how estimates improve with more simulations  
- **Terminal Distribution**: Histogram of final prices at expiration  
- **Heston Model Integration**: Simulate stochastic volatility paths
  
  ![MC](https://github.com/user-attachments/assets/b683f027-dd6f-4d7a-84dc-f1293f30e4b4)
  ![11](https://github.com/user-attachments/assets/ffd4e851-0483-410a-8ef8-d0d66408b9a8)
  ![10](https://github.com/user-attachments/assets/89e10b31-6f8f-40e5-8db9-014f246c06d4)





### Volatility Analysis Suite
- **Implied Volatility Calculator**: Extract IV from market prices  
- **Volatility Smile Visualization**: Display skew across strikes  
- **Term Structure Analysis**: IV across different maturities  
- **3D Volatility Surface**: Interactive surface plot of entire vol surface

  ![15](https://github.com/user-attachments/assets/c3812f51-952f-418b-b9ae-e08fef4a8852)
  ![13](https://github.com/user-attachments/assets/06e2ec23-9b0c-4265-a318-86eae00906c2)
  ![14](https://github.com/user-attachments/assets/387a5b92-ad42-4f9f-8dce-b09e34fc1f02)




### Interactive Option Chain
- **Real-time Greeks**: All Greeks displayed for each strike  
- **Volume & Open Interest**: Visual representation of market activity  
- **Max Pain Analysis**: Calculate and visualize max pain point  
- **Put/Call Ratios**: Sentiment indicators based on volume and open interest  
- **Advanced Filtering**: Filter by moneyness, volume, and other criteria  

![17](https://github.com/user-attachments/assets/c26f1097-bf2c-4f80-a86f-adb017cddcfe)
![18](https://github.com/user-attachments/assets/442065b0-b5e7-425c-8c9d-811658ed1da8)


---

## Technical Implementation

### Technologies Used
- **Python 3.8+**: Core programming language  
- **Streamlit**: Web application framework  
- **NumPy**: Numerical computations and array operations  
- **Pandas**: Data manipulation and analysis  
- **Plotly**: Interactive visualizations and 3D plots  
- **SciPy**: Statistical functions and optimization algorithms  

### Mathematical Models
- Closed-form solutions for Black-Scholes  
- Binomial Tree numerical methods for American options  
- Monte Carlo methods with variance reduction  
- Finite-difference methods for Greeks calculations  
- Newton-Raphson solver for implied volatility  

### Design Features
- **Professional Black Theme**: High-contrast design for better visibility  
- **Responsive Layout**: Adapts to different screen sizes  
- **Interactive Controls**: Real-time parameter updates  
- **Export Capabilities**: Download results and visualizations  

---

## Installation & Usage

### Prerequisites
- Python 3.8 or higher  
- `pip` package manager  

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/option-pricing-tool.git
cd option-pricing-tool

# Install required packages
pip install -r requirements.txt
```
### Running the Application
```bash
streamlit run option_pricing_tool.py
```

---

## Usage Guide
1. **Set Global Parameters**
- Use the sidebar to input spot price, strike price, time to maturity, etc.

- Adjust model-specific parameters in expandable sections

2. **Price Options**

- Navigate to the Option Pricing tab

- Select option type (Call/Put) and exercise style

3. **Click Calculate Prices to see results across all models**

- Analyze Greeks

- Switch to the Greeks Analysis tab

- View 2D plots showing how Greeks change with spot price

- Explore 3D surface plots for multi-dimensional analysis

3. **Build Strategies**

- Go to the Strategy Builder tab

- Select predefined strategies or build custom ones

- Analyze payoff diagrams with profit/loss zones

4. **Run Simulations**

- Open the Monte Carlo tab for path-dependent analysis

- Adjust the number of simulations and paths displayed

- View convergence and terminal distribution plots

## Contributing
**Contributions are welcome!**

1. Fork the repository

2. Create your feature branch `git checkout -b feature/YourFeature`

3. Commit your changes `git commit -m "Add YourFeature"`

4. Push to the branch `git push origin feature/YourFeature`

5. Open a Pull Request

For major changes, please open an issue first to discuss.

## License
This project is licensed under the MIT License.

## Author
- Arnav Jain

- LinkedIn: [arnavj19](https://www.linkedin.com/in/arnavj19) 

- GitHub: [@ArnavJ19](https://github.com/ArnavJ19)

- Portfolio: [Click Here](https://www.arnavxdata.com)

## Acknowledgments
- Inspired by professional trading platforms and academic research
- Built with open-source technologies and community support
