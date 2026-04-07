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
The score must be an integer from 1 to 5 (never 0).
Respond with this exact JSON format:
{{"dimension": "{dimension}", "score": <1-5>, "explanation": "<brief explanation>"}}"""


GEVAL_STEP_GENERATION_SYSTEM_PROMPT = """\
You are an evaluation methodology expert. Given an evaluation dimension and \
its rubric, generate a concise list of 3-6 concrete evaluation steps that a \
judge should follow to score an AI agent's output on this dimension.

Each step should be specific, actionable, and directly tied to the rubric criteria.
Respond with a JSON array of strings, no additional text."""

GEVAL_STEP_GENERATION_USER_TEMPLATE = """\
## Evaluation Dimension
{dimension}

## Rubric
{rubric_text}

## Instructions
Generate 3-6 evaluation steps as a JSON array of strings.
Example format: ["Step 1 description", "Step 2 description", ...]"""

GEVAL_SCORING_SYSTEM_PROMPT = """\
You are an evaluation judge for AI agent behavior. You will be given:
1. The original query given to the agent
2. The agent's trajectory (tool calls and observations)
3. The agent's final answer
4. A reference answer
5. An evaluation rubric with structured evaluation steps

Follow the evaluation steps one by one. After completing all steps, \
produce your final score.
Respond with valid JSON only, no additional text."""

GEVAL_SCORING_USER_TEMPLATE = """\
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

## Evaluation Steps
Follow these steps sequentially to evaluate the output:
{evaluation_steps}

## Instructions
After following all evaluation steps above, score the agent on the \
"{dimension}" dimension (1-5).
The score must be an integer from 1 to 5 (never 0).
Respond with this exact JSON format:
{{"dimension": "{dimension}", "score": <1-5>, "explanation": "<brief explanation referencing the evaluation steps>"}}"""


TASK_EXTRACTION_SYSTEM_PROMPT = """\
You are a task analysis expert. Given a user query to an AI agent, decompose \
it into discrete sub-tasks that the agent must complete. Each sub-task should \
be independently verifiable.

Respond with a JSON array of strings, no additional text."""

TASK_EXTRACTION_USER_TEMPLATE = """\
## User Query
{query}

## Instructions
Decompose the query into discrete sub-tasks. Each sub-task should be a single, \
clear objective that can be independently verified as completed or not.
Respond as a JSON array: ["sub-task 1", "sub-task 2", ...]"""

TASK_COMPLETION_SYSTEM_PROMPT = """\
You are an evaluation judge for AI agent task completion. You will be given:
1. A list of sub-tasks the agent was expected to complete
2. The agent's execution trajectory
3. The agent's final answer

For each sub-task, determine whether it was completed successfully.
Respond with valid JSON only, no additional text."""

TASK_COMPLETION_USER_TEMPLATE = """\
## Sub-Tasks
{tasks}

## Agent Trajectory
{trajectory}

## Final Answer
{final_answer}

## Instructions
For each sub-task, judge whether it was completed based on the trajectory \
and final answer. Respond with this exact JSON format:
{{"tasks": [{{"task": "<task description>", "completed": true/false, "evidence": "<brief evidence>"}}]}}"""


STATEMENT_EXTRACTION_SYSTEM_PROMPT = """\
You are a text analysis expert. Given a text, decompose it into independent \
atomic statements. Each statement should express exactly one fact or claim.

Respond with a JSON array of strings, no additional text."""

STATEMENT_EXTRACTION_USER_TEMPLATE = """\
## Text
{answer_text}

## Instructions
Decompose the text into atomic statements. Each statement should be a single, \
self-contained factual claim or assertion.
Respond as a JSON array: ["statement 1", "statement 2", ...]"""

STATEMENT_RELEVANCE_SYSTEM_PROMPT = """\
You are a relevance judge. Given a user query and a list of statements \
extracted from an agent's answer, determine whether each statement is \
relevant to answering the query.

A statement is relevant if it directly addresses, supports, or is necessary \
for answering the query. Tangential or off-topic statements are not relevant.

Respond with valid JSON only, no additional text."""

STATEMENT_RELEVANCE_USER_TEMPLATE = """\
## User Query
{query}

## Statements
{statements}

## Instructions
For each statement, judge whether it is relevant to the query.
Respond with this exact JSON format:
{{"results": [{{"statement": "<statement>", "relevant": true/false}}]}}"""


CONTRADICTION_DETECTION_SYSTEM_PROMPT = """\
You are a factual consistency judge. Given an agent's output and a list of \
context items (source facts), determine whether the output contradicts any \
of the context items.

A contradiction exists when the output makes a claim that directly conflicts \
with or negates a context item. Omission (not mentioning a context item) is \
NOT a contradiction.

Respond with valid JSON only, no additional text."""

CONTRADICTION_DETECTION_USER_TEMPLATE = """\
## Agent Output
{output_text}

## Context Items
{context_items}

## Instructions
For each context item, determine whether the agent's output contradicts it.
Remember: omission is NOT contradiction. Only flag direct conflicts.
Respond with this exact JSON format:
{{"results": [{{"context": "<context item>", "contradicted": true/false, "explanation": "<brief explanation>"}}]}}"""


SUPPORT_DETECTION_SYSTEM_PROMPT = """\
You are a factual support judge. Given a list of statements from an agent's \
answer and a list of context items, determine whether each statement is \
supported by the provided context.

A statement is supported only when the context directly backs it. If the \
context is missing the needed evidence, mark it as unsupported. Do not infer \
extra facts beyond the context.

Respond with valid JSON only, no additional text."""

SUPPORT_DETECTION_USER_TEMPLATE = """\
## Statements
{statements}

## Context Items
{context_items}

## Instructions
For each statement, determine whether it is supported by the context.
Respond with this exact JSON format:
{{"results": [{{"statement": "<statement>", "supported": true/false, "explanation": "<brief explanation>"}}]}}"""
