## Methodology

The debate pipeline implements a multi-agent debate system(3 agents), which includes two debater agents and one judge. Debater A (Llama 3.1 8B) and Debater B (GPT OSS 20B) are assigned opposing positions on a multiple-choice question and argue their case towards multiple rounds. The judge (Qwen3-8B) observes the full debate transcript and renders a structured verdict.

The Llama 3.1 8B model was selected because it provides strong reasoning performance while remaining computationally efficient. Its relatively smaller parameter size allows faster inference and enables multiple debate rounds without excessive resource consumption. The GPT-OSS 20B model was chosen to provide a debater with higher model capacity and potentially more complex reasoning capabilities. By pairing a larger model with a smaller model, the debate system can produce richer argumentative interactions and reduce the likelihood of identical reasoning paths.

The judge agent uses Qwen3 8B, which was selected for its strong performance in reasoning and evaluation tasks. The judge's role differs from that of the debaters. Rather than generating new arguments, the judge must analyze the entire debate transcript and determine which argument is more sound. The Qwen3 model was chosen because it demonstrates reliable reasoning and summarization capabilities, making it well suited for evaluating competing arguments.

## Debate Protocol

- A question is provided to both debaters.
- Debater A is given a side to argue for.
- Debater B tries to give a counterargument supporting the other option.
- The debate goes on for N Rounds where the agents (both debaters) criticize each other's reasoning and refine their argument.
- The judge looks at the transcript of the debate, analyzing both points and coming up with a verdict i.e. the winner of the debate.

  ![Results](https://raw.githubusercontent.com/tobi-olad/llm-debate-pipeline/main/img.png))

## Model Configurations

The temperature of the debaters was chosen to be 0.7 while the temperature of the judge was chosen to be 0.3. I opted for the debaters to have that temperature to give the idea of the agents or debaters in this case to creatively think of ways that they can attack each other's points instead of a very deterministic way of arguing, while for the judge I gave it a low temperature because a judge is supposed to be a little bit more deterministic.

| Agent     | Model        | Temperature | Max Tokens |
| --------- | ------------ | ----------- | ---------- |
| Debater 1 | Llama 3.1 8B | 0.7         | 1024       |
| Debater 2 | GPT OSS 20B  | 0.7         | 1024       |
| Judge     | Qwen3-8B     | 0.3         | 1536       |

The debater agents were assigned a maximum generation length of 1024 tokens to allow sufficient space for structured arguments while preventing excessively long responses. The debaters must adhere to the following principles:

- Present their opinions and reasoning
- Critique the opponent
- Provide rebuttals
- Summarize their final stance

Using larger limits may encourage illogical reasoning and may cause the argument quality to be reduced.

The judge is given a larger max token limit because it must process the entire debate transcript before coming to a final decision. Unlike the debaters, the judge does the following:

- Analyze arguments from both debaters
- Compare Reasoning Quality
- Identify Logical Strength and weaknesses
- Produce a final verdict

Since the judge receives multiple rounds of arguments as input, additional generation ensures the model can fully evaluate the debate and provide a detailed reasoning process without being truncated.

The experiments are conducted on the ARC-Challenge benchmark (Clark et Al, 2018), a multiple-choice question and answer dataset. A sample of 100 questions is taken from the test split with a random seed of 42 (each with 4 answer choices A-D).

All three methods — the debate, the Direct QA, and the self-consistency — are evaluated on the same 100 questions to ensure fairness. Each method uses the Llama 3.1-8B-Instruct Model as a primary answering agent. The self-consistency samples 5 independent answers at a temperature of 0.9 and takes the majority vote, which matches the approximate number of calls used in one debate run. Direct QA used a single zero-shot chain of thought prompt at 0.7.

## Table of Results

The QA baseline achieves an accuracy of 82%, which represents the model's performance when answering questions directly without additional reasoning or verification steps. In this setting, the model receives the question once and produces a single answer. Because only one inference call is made, this method is computationally efficient. However, it lacks mechanisms to detect reasoning errors or reconsider incorrect answers. As a result, the model may confidently produce incorrect responses, leading to lower accuracy compared to the other approaches.

The self-consistency approach improves accuracy slightly to 85% by generating multiple independent reasoning paths and selecting the most common answer. This method attempts to reduce reasoning errors by aggregating multiple outputs from the model. By sampling several reasoning chains (approximately five in this experiment), the system can filter out occasional mistakes made in individual responses. However, because all answers originate from the same model, the reasoning patterns may still be highly correlated. This limits the degree of improvement over the baseline QA approach. Additionally, self-consistency requires multiple LLM calls, increasing computational cost relative to simple QA.

The debate-based approach achieves the highest accuracy at 93%, significantly outperforming both QA and self-consistency. This improvement can be attributed to the adversarial reasoning structure of the debate framework. Instead of generating multiple independent answers, two debater agents actively critique each other's arguments across several rounds. This interaction helps expose logical errors, weak assumptions, and unsupported claims that might otherwise go unnoticed in a single-pass response. The judge model then evaluates the debate transcript and selects the argument that demonstrates stronger reasoning. This process allows the system to incorporate cross-examination and argument evaluation, which can improve decision quality. However, this increased accuracy comes at the cost of additional computation. Each debate involves multiple model calls for argument generation, rebuttals, and judging, resulting in approximately 8–10 LLM calls per question.

| Method      | Accuracy | Correct | N   | Avg LLM Calls |
| ----------- | -------- | ------- | --- | ------------- |
| Debate      | 93%      | 93%     | 100 | ~8-10         |
| QA          | 82%      | 84%     | 100 | 1             |
| Consistency | 85%      | 84%     | 100 | 5             |

## Analysis

Examine four representative debates: two where the debate produced the correct answer, two where it failed, and an early stop case.

**Transcript 1 — Debaters disagreed and judge predicted correctly**

Question 20: David makes a solution by dissolving 10 grams of salt in 100 ml of water. He wants a solution that is half as concentrated. What should he add to the original solution to obtain a solution that is about half as concentrated?

Initial Positions: Debater A: 5 grams of salt | Debater B: 100 ml of water. Ground Truth: 100 ml of water.

In this case Debater B's argument is based on mathematical fact while Debater A's argument is based on a hypothetical. The judge agent favors a factual based argument rather than a hypothetical, which is why it chooses Debater B as winner with a confidence of 5/5.

**Transcript 2 — Early Stop**

Question 47: Which change in Earth's surface is most directly related to the water cycle?

Initial Positions: Debater A: movement of tectonic plates | Debater B: deposition of sediments. Ground Truth: deposition of sediments.

The debate goes on for 2 rounds before they come to a consensus. Debater B provided a fact that was irrefutable, leading to a shift in Debater A's position and triggering an early stop.

**Transcript 3 — Debate Ran, Judge was wrong**

Question 31: Beavers build their homes in ponds and streams. Which characteristic is least critical to building homes in an aquatic environment?

Initial Positions: Debater A: waterproof fur | Debater B: flat, wide tail. Ground Truth: C.

Both debaters made plausible but factually incorrect arguments. Neither cited specific evidence and both relied on surface-level reasoning. The judge awarded the winner based on structure rather than factual accuracy, leading to the wrong verdict.

**Transcript 4 — Consensus, both wrong**

Question 8: Which tool would be the most helpful in an investigation of the life cycle of a monarch butterfly?

Initial Positions: both agreed on magnifying glass. Ground Truth: a large jar with air holes in the top.

This is a limitation of using consensus to pick an option. It is entirely possible that the models share the same bias and the same knowledge gap, making it easy for the models to come to the same conclusion even though they are wrong.

## Prompt Engineering Process

Each agent uses a structured prompt template with explicit output format constraints. Debater A is instructed to state a position, provide step-by-step reasoning, deliver a REBUTTAL argument targeting the opponent's weakest point, and close with a SUMMARY. Debater B follows the same structure with an additional ATTACK field targeting Debater A's main flaw. The judge prompt requires COT_ANALYSIS, STRONGEST AND WEAKEST argument fields for each side, WINNER, VERDICT, CONFIDENCE, and REASONING. All responses are capped at 350 words for debaters and 500 words for the judge to encourage conciseness.

## Future Work

Prompt design plays a critical role in guiding agent behavior. Future work could experiment with stronger adversarial prompts that explicitly require debaters to identify logical flaws in opposing arguments. Additionally, structured prompting techniques such as argument trees or evidence citation may improve reasoning quality and reduce hallucinated information.

The accuracy of the final decision also depends heavily on the judge model. In some cases, the judge selected persuasive but incorrect arguments. Future work could investigate alternative judging strategies, such as ensemble judges, majority voting across multiple judge models, or step-by-step verification of arguments.
