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

- Python 3.8+
- OpenAI API Key

### 2. Install Dependencies

#### Install OASIS Framework

According to the [OASIS official documentation](https://github.com/camel-ai/oasis), first install the OASIS package:

```bash
pip install camel-oasis
```

#### Install Project Dependencies

```bash
pip install python-dotenv
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
# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Specify OpenAI model
OPENAI_MODEL=gpt-4o-mini

# Database Configuration
DATABASE_PATH=market_sim.db

# Simulation Parameters
TOTAL_AGENTS=6
NUM_SELLERS=10
NUM_BUYERS=10
SIMULATION_ROUNDS=7
```

#### Environment Variables Reference

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `DATABASE_PATH` | Database file path | `market_sim.db` |
| `TOTAL_AGENTS` | Total number of agents | `6` |
| `NUM_SELLERS` | Number of sellers | `10` |
| `NUM_BUYERS` | Number of buyers | `10` |
| `SIMULATION_ROUNDS` | Number of simulation rounds | `7` |

### 4. Run Simulation

```bash
python run_market_simulation.py
```

## 🛠️ Customization

### Modify Agent Count

Adjust in `run_market_simulation.py`:

```python
TOTAL_AGENTS = 6      # Total number of agents
NUM_SELLERS = 10      # Number of sellers
NUM_BUYERS = 10       # Number of buyers
SIMULATION_ROUNDS = 7 # Number of simulation rounds
```

### Customize Agent Characteristics

Modify prompt templates in `prompt.py`:

- `SELLER_GENERATION_SYS_PROMPT`: Seller generation system prompt
- `SELLER_GENERATION_USER_PROMPT`: Seller generation user prompt
- `BUYER_GENERATION_SYS_PROMPT`: Buyer generation system prompt
- `BUYER_GENERATION_USER_PROMPT`: Buyer generation user prompt

### Adjust Market Parameters

Modify market-related parameters in the code:

```python
# Product price range
PRICE_RANGE = (10, 50)

# Warranty cost
WARRANT_ESCROW = 5

# Reputation update weight
REPUTATION_WEIGHT = 0.1
```

## 📚 Related Resources

- [OASIS Official Documentation](https://docs.oasis.camel-ai.org/)
- [OASIS GitHub Repository](https://github.com/camel-ai/oasis)
- [CAMEL-AI Project](https://github.com/camel-ai/camel)

## 🙏 Acknowledgments

Thanks to the [OASIS](https://github.com/camel-ai/oasis) project for providing an excellent multi-agent simulation framework, and to the [CAMEL-AI](https://github.com/camel-ai/camel) team for their important contributions in the AI agent field.