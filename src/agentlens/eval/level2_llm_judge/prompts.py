"""Prompt templates for LLM-as-Judge evaluation."""

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for AI agent behavior. You will be given:
1. The original query given to the agent
2. The agent's trajectory (tool calls and observations)
3. The agent's final answer
4. A reference answer
5. An evaluation rubric

Score the agent on the specified dimension using the rubric (1-5 scale).
Respond with valid JSON only, no additional text."""

JUDGE_USER_TEMPLATE = """\
## Query
{query}

## Agent Trajectory
{trajectory}

## Final Answer
{final_answer}

## Reference Answer
{reference_answer}

## Evaluation Rubric
Dimension: {dimension}
{rubric_text}

## Instructions
Score the agent on the "{dimension}" dimension (1-5).
Respond with this exact JSON format:
{{"dimension": "{dimension}", "score": <1-5>, "explanation": "<brief explanation>"}}"""
