# aws-bedrock-langgraph

Agent built on AWS Bedrock using Langgraph framework

## Quick Start Guide

### Prerequisites

- Python 3.13 or higher
- AWS Account with Bedrock access
- Required Python packages (see `requirements.txt`)

### Installation

1. Install the project dependencies:
```bash
pip install -r requirements.txt
```

## Setup Process

### Step 1: Configure Environment Variables

Create a `.env.local` file in the project root directory with the following required variables:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default

# Bedrock Model Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Agent Configuration
AGENT_NAME=bedrock-langgraph-agent
AGENT_DESCRIPTION=AWS Bedrock Agent with Langgraph Framework

# Optional: Groq API (if using Groq models)
GROQ_API_KEY=your-groq-api-key-here

# Optional: HuggingFace Configuration
HUGGINGFACE_API_TOKEN=your-huggingface-token-here
```

**Important Variables:**
- `AWS_REGION`: The AWS region where your Bedrock service is available
- `AWS_PROFILE`: Your AWS credentials profile (default or custom)
- `BEDROCK_MODEL_ID`: The model ID from AWS Bedrock to use for your agent

### Step 2: Configure Agentcore

Run the agentcore configuration command to initialize your agent:

```bash
agentcore configure
```

This will:
- Validate your AWS credentials
- Test connectivity to Bedrock
- Initialize the agent configuration
- Set up checkpoint storage if needed

### Step 3: Deploy the Agent

Deploy your agent using:

```bash
agentcore deploy
```

This will:
- Package your agent code
- Deploy to your configured environment
- Set up any necessary AWS resources (IAM roles, S3 buckets, etc.)
- Provide deployment status and endpoints

### Step 4: Invoke the Agent

Once deployed, you can invoke your agent with:

```bash
agentcore invoke "{'prompt': 'Tell me about roaming activations'}"
```

**Example Prompt:**
```bash
agentcore invoke --input "Based on the data in lauki_qna.csv, what are the top 3 questions asked by users?"
```

This will:
- Send your prompt to the deployed agent
- Process it through the Bedrock model
- Return the agent's response with any tool outputs

## Project Structure

```
.
├── agentcore_memory.py          # Memory management configuration
├── agentcore_runtime.py         # Runtime and execution logic
├── lauki_qna.csv               # Sample Q&A data
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── aws/                       # AWS-specific configurations
    └── README.md             # AWS setup details
```

## Troubleshooting

- **AWS Credentials Error**: Ensure your AWS profile is configured correctly using `aws configure`
- **Bedrock Access**: Verify that your AWS account has access to the Bedrock service
- **Environment Variables**: Confirm all required variables are set in `.env.local`
- **Deployment Issues**: Check AWS CloudWatch logs for detailed error messages

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Bedrock AgentCore Documentation](https://aws-bedrock-langgraph)
