"""
Evaluate LoRA-based Persona Steering

This script evaluates models with LoRA-based steering, providing the weight-space
analog to activation-based steering from the original paper.

Key difference: Instead of adding to activations (h + α·v), we merge LoRA weights (W + α·ΔW_lora)

Usage:
    # Baseline (no steering)
    python eval_lora_persona.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --trait evil \
        --output_path results/baseline.csv

    # With LoRA steering
    python eval_lora_persona.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --trait evil \
        --lora_diff_path persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt \
        --lora_coef 2.0 \
        --output_path results/lora_steering.csv
"""

import os
import asyncio
import json
import torch
import pandas as pd
import random
from typing import Dict, List, Optional
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge import OpenAiJudge
from eval.prompts import Prompts
from config import setup_credentials
from lora_utils import LoRASteerer, LoRAMerger

config = setup_credentials()


def sample_with_lora_steering(
    model,
    tokenizer,
    conversations: List[Dict],
    lora_diff_path: str,
    coef: float,
    bs: int = 20,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1
):
    """
    Generate responses with LoRA-based steering.

    Merges the LoRA difference into the model before generation.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare prompts
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Apply LoRA steering
    print(f"Applying LoRA steering with coefficient α={coef}...")
    merger = LoRAMerger(model)
    lora_diff = torch.load(lora_diff_path, map_location='cpu', weights_only=False)
    merger.merge_lora_diff(lora_diff, coefficient=coef)

    # Generate with steering
    outputs = []
    try:
        for i in trange(0, len(prompts), bs, desc="Generating with LoRA steering"):
            batch = prompts[i:i+bs]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
            tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens
                )

            prompt_len = tokenized_batch["input_ids"].shape[1]
            output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
            outputs.extend(output)

    finally:
        # Always restore original weights
        merger.restore_original_weights()
        print("✓ Restored original weights")

    return prompts, outputs


def sample_baseline(
    model,
    tokenizer,
    conversations: List[Dict],
    bs: int = 20,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1
):
    """Generate responses without steering (baseline)."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating baseline"):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        with torch.no_grad():
            output = model.generate(
                **tokenized_batch,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                min_new_tokens=min_tokens
            )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)

    return prompts, outputs


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait: str,
    temperature: float = 1.0,
    persona_instructions_type: Optional[str] = None,
    assistant_name: Optional[str] = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    eval_type: str = "0_100",
    version: str = "eval"
):
    """Load evaluation questions for a trait."""
    from eval.eval_persona import Question

    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json", "r"))

    judge_prompts = {}
    judge_prompts[trait] = trait_data["eval_prompt"]
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]

    raw_questions = trait_data["questions"]
    questions = []

    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                if assistant_name is None:
                    assistant_name = trait if persona_instructions_type == "pos" else "helpful"

                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
                questions.append(Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    system=system,
                    judge_eval_type=eval_type
                ))
        else:
            questions.append(Question(
                paraphrases=[question],
                id=f"{trait}_{i}",
                judge_prompts=judge_prompts,
                judge=judge_model,
                temperature=temperature,
                judge_eval_type=eval_type
            ))

    return questions


async def eval_batched_lora(
    questions,
    model,
    tokenizer,
    lora_diff_path: Optional[str] = None,
    lora_coef: float = 0.0,
    n_per_question: int = 100,
    max_concurrent_judges: int = 100,
    max_tokens: int = 1000
):
    """Batch process all questions with LoRA steering."""
    # Collect all prompts
    all_paraphrases = []
    all_conversations = []
    question_indices = []

    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    # Generate all answers
    print(f"Generating {len(all_conversations)} responses...")
    if lora_diff_path and lora_coef != 0:
        prompts, answers = sample_with_lora_steering(
            model,
            tokenizer,
            all_conversations,
            lora_diff_path,
            lora_coef,
            temperature=questions[0].temperature,
            max_tokens=max_tokens
        )
    else:
        prompts, answers = sample_baseline(
            model,
            tokenizer,
            all_conversations,
            temperature=questions[0].temperature,
            max_tokens=max_tokens
        )

    # Prepare for judge evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        df = pd.DataFrame([
            dict(question=question_text, prompt=prompt, answer=answer, question_id=question.id)
            for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
        ])
        question_dfs.append(df)

        # Collect judge tasks
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    # Run judges with concurrency control
    print(f"Running {len(all_judge_tasks)} judge evaluations...")
    semaphore = asyncio.Semaphore(max_concurrent_judges)
    all_results = [None] * len(all_judge_tasks)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result
            pbar.update(1)

    # Distribute results
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def main(
    model: str,
    trait: str,
    output_path: str,
    lora_diff_path: Optional[str] = None,
    lora_coef: float = 0.0,
    max_tokens: int = 1000,
    n_per_question: int = 10,
    max_concurrent_judges: int = 100,
    persona_instruction_type: Optional[str] = None,
    assistant_name: Optional[str] = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    version: str = "eval",
    overwrite: bool = False
):
    """Evaluate a model with optional LoRA steering."""
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, loading...")
        df = pd.read_csv(output_path)
        for metric in [trait, "coherence"]:
            print(f"{metric}: {df[metric].mean():.2f} ± {df[metric].std():.2f}")
        return

    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temperature = 0.0 if n_per_question == 1 else 1.0

    # Load model
    print(f"Loading model: {model}")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load questions
    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version
    )

    print(f"Evaluating {len(questions)} questions for trait '{trait}'...")

    # Run evaluation
    outputs_list = asyncio.run(eval_batched_lora(
        questions,
        model_obj,
        tokenizer,
        lora_diff_path=lora_diff_path,
        lora_coef=lora_coef,
        n_per_question=n_per_question,
        max_concurrent_judges=max_concurrent_judges,
        max_tokens=max_tokens
    ))

    outputs = pd.concat(outputs_list)
    outputs.to_csv(output_path, index=False)

    print(f"\n✓ Results saved to {output_path}")
    for metric in [trait, "coherence"]:
        print(f"{metric}: {outputs[metric].mean():.2f} ± {outputs[metric].std():.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
