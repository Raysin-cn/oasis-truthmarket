# OASIS Truth Market Simulation

A multi-agent online market simulation system built on the [OASIS](https://github.com/camel-ai/oasis) framework, designed to study agent behavior patterns in realistic market environments, with a particular focus on information asymmetry, reputation mechanisms, and warranty systems' impact on market efficiency.

## 🎯 Overview

This project implements a multi-agent online market simulation environment featuring:

- **Seller Agents**: Can list high or low quality products, choose whether to offer warranties
- **Buyer Agents**: Make purchasing decisions based on seller reputation and product information
- **Market Mechanisms**: Include reputation systems, warranty institutions, and transaction tracking
- **Data Analysis**: Real-time statistics and market performance visualization

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+ (< 3.12)
- OpenAI API Key

### 2. Install Dependencies

#### Install OASIS Framework

According to the [OASIS official documentation](https://github.com/camel-ai/oasis), first install the OASIS package:

```bash
pip install camel-oasis
```

#### Install Project Dependencies

```bash
pip install -r requirement.txt
```

### 3. Environment Configuration

#### Create Environment Variables File

The project provides an `env.template` file as a configuration template. Please follow these steps:

1. Copy the template file:
```bash
cp env.template .env
```

2. Edit the `.env` file with your actual configuration:
```bash
# Edit .env file
nano .env  # or use your preferred editor
```

3. At minimum, configure the following required items:
```bash
# Model API Configuration (Required)
MODEL_API_KEY=your_api_key_here
MODEL_BASE_URL=https://api.openai.com/v1  # Optional, for OpenAI or compatible APIs

# Optional: Database path
MARKET_DB_PATH=market_sim.db
```

#### Environment Variables Reference

| Variable | Description | Default Value | Required |
|----------|-------------|---------------|----------|
| `MODEL_API_KEY` | API key for the model provider (OpenAI or vLLM) | - | ✓ Yes |
| `MODEL_BASE_URL` | Base URL for API endpoint (for custom endpoints) | - | Optional |
| `MARKET_DB_PATH` | Database file path for market simulation | `market_sim.db` | Optional |

#### Configuration via config.py

The simulation parameters are configured in `config.py`:

```python
class SimulationConfig:
    RUNS = 50                          # Total number of independent simulation runs
    NUM_SELLERS = 10                   # Number of seller agents per run
    NUM_BUYERS = 10                    # Number of buyer agents per run
    SIMULATION_ROUNDS = 7              # Number of trading rounds per run
    
    # Market mechanism parameters
    REPUTATION_LAG = 1                 # Rounds of delay in reputation display
    REENTRY_ALLOWED_ROUND = 5          # Round when low-reputation sellers can re-enter
    EXIT_ROUND = 7                     # Round when sellers can choose to exit
    MARKET_TYPE = 'reputation_and_warrant'  # Market mechanism type
    
    # Model configuration
    MODEL_PLATFORM = "openai"          # "openai" or "vllm"
    MODEL_TYPE = "gpt-4o"              # Model identifier
```

To modify simulation parameters, edit these values in `config.py` before running.

### 4. Run Simulation

#### Option 1: Run Multiple Independent Experiments (Recommended)

To run multiple independent market simulations with data collection and aggregated analysis:

```bash
python run_experiments.py
```

This will:
- Run the number of simulations specified by `SimulationConfig.RUNS` (default: 50)
- Create an experiment directory with a timestamp-based ID
- Save individual run databases
- Collect and analyze results from all runs
- Generate aggregated analysis reports

#### Option 2: Run Single Simulation

To run a single market simulation:

```bash
# Run with custom database path
python run_single_simulation.py test_run.db

# Run with default database path
python run_single_simulation.py
```

This creates a standalone simulation run with results saved to the specified database file.

## 🛠️ Customization

### Modify Simulation Parameters

Edit values in `config.py`:

```python
class SimulationConfig:
    RUNS = 50                          # Total number of independent simulation runs
    NUM_SELLERS = 10                   # Number of seller agents per run
    NUM_BUYERS = 10                    # Number of buyer agents per run
    SIMULATION_ROUNDS = 7              # Number of trading rounds per run
    
    # Market mechanism parameters
    REPUTATION_LAG = 1                 # Rounds of delay in reputation display
    REENTRY_ALLOWED_ROUND = 5          # Round when low-reputation sellers can re-enter
    EXIT_ROUND = 7                     # Round when sellers can choose to exit
    MARKET_TYPE = 'reputation_and_warrant'  # Market mechanism type
    
    # Model configuration
    MODEL_PLATFORM = "openai"          # "openai" or "vllm"
    MODEL_TYPE = "gpt-4o"              # Model identifier
```

### Customize Agent Characteristics

Modify prompt templates in `prompt.py`:

- `SELLER_GENERATION_SYS_PROMPT`: System prompt for seller agent generation
- `SELLER_GENERATION_USER_PROMPT`: User prompt for seller agent generation
- `BUYER_GENERATION_SYS_PROMPT`: System prompt for buyer agent generation
- `BUYER_GENERATION_USER_PROMPT`: User prompt for buyer agent generation
- `SELLER_ROUND_PROMPT`: Dynamic prompt for sellers during each round
- `BUYER_ROUND_PROMPT`: Dynamic prompt for buyers during each round

### Adjust Market Parameters

## 📊 Data Analysis

After running simulations, use the analysis tools to generate visualizations and insights:

### 1. Single Run Analysis

Analyze a single market simulation database:

```bash
python analysis/analyze_market.py <database_path> [--out output_directory]
```

**Examples:**
```bash
# Analyze single run with default output directory
python analysis/analyze_market.py experiments/experiment_20251030_120000/run_1.db

# Specify custom output directory
python analysis/analyze_market.py test_run.db --out my_analysis_output
```

**Generated visualizations:**
- `reputation_over_rounds.png` - Seller reputation evolution across rounds
- `avg_price_by_advertised_quality.png` - Average listing prices by quality level
- `seller_actions_scatter.png` - Seller actions (listing, exit, re-entry) by round
- `manipulation_behavior_statistics.png` - Manipulation behavior patterns
- `seller_manipulation_details.png` - Detailed manipulation analysis per seller

### 2. Multi-Run Aggregated Analysis

Analyze all runs from a completed experiment:

```bash
python analysis/multi_run_analysis.py --experiment_id <experiment_id>
```

**Example:**
```bash
python analysis/multi_run_analysis.py --experiment_id experiment_20251030_120000
```

**Features:**
- Cross-run comparison analysis
- Round-by-round progression trends
- Distribution analysis across all runs
- Seller deception behavior analysis
- Individual run analysis for each simulation

**Generated outputs:**
- `analysis/<experiment_id>/aggregated/aggregated_statistics.json` - Aggregated metrics
- `analysis/<experiment_id>/aggregated/` - Visualizations
  - `cross_run_comparison.png` - Comparison across all runs
  - `round_progression.png` - Trends by round
  - `distribution_analysis.png` - Distribution patterns
  - `seller_deception_analysis.png` - Deception behavior analysis
- `analysis/<experiment_id>/individual_runs/` - Analysis for each run

### 3. Market Mechanism Comparison

Compare two market mechanisms (Reputation-Only vs Reputation+Warrant):

```bash
python analysis/create_comparison_charts.py
```

**Setup required:**
Edit the `EXPERIMENT_CONFIG` in `create_comparison_charts.py` to specify your experiment IDs:

```python
EXPERIMENT_CONFIG = {
    'reputation_only': "your_reputation_only_experiment_id",
    'reputation_warrant': "your_reputation_warrant_experiment_id"
}
```

**Generated comparison charts:**
- `market_mechanism_comparison_summary.png` - Core metrics (buyer utility, seller profit, transactions)
- `market_mechanism_round_progression.png` - Round-by-round progression comparison
- `market_mechanism_distribution_comparison.png` - Distribution analysis
- `market_mechanism_deception_behavior.png` - Deception behavior comparison
- `config.json` - Configuration and statistics metadata

**Analysis coverage:**
- Average buyer utility per run
- Average seller profit per run
- Transaction volumes
- Deception behavior (HQ advertised but LQ delivered)
- Statistical distributions and trends

## 📊 Analysis Workflow Example

```bash
# 1. Run multiple experiments
python run_experiments.py --runs 50

# 2. Get experiment ID from output (e.g., experiment_20251030_120000)

# 3. Analyze aggregated results
python analysis/multi_run_analysis.py --experiment_id experiment_20251030_120000

# 4. View results in analysis/experiment_20251030_120000/aggregated/

# 5. For mechanism comparison (if you have two experiments to compare)
python analysis/create_comparison_charts.py
```

## 📚 Related Resources

- [OASIS Official Documentation](https://docs.oasis.camel-ai.org/)
- [OASIS GitHub Repository](https://github.com/camel-ai/oasis)
- [CAMEL-AI Project](https://github.com/camel-ai/camel)

## 🙏 Acknowledgments

Thanks to the [OASIS](https://github.com/camel-ai/oasis) project for providing an excellent multi-agent simulation framework, and to the [CAMEL-AI](https://github.com/camel-ai/camel) team for their important contributions in the AI agent field.
