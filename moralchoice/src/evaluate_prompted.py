import json
from typing import List, Dict
from src.question_form_generator import get_question_form
from src.semantic_matching import token_to_action_matching

# Load refusals and common answer patterns
with open(f"data/response_templates/refusals.txt", encoding="utf-8") as f:
    refusals = f.read().splitlines()

response_patterns = {}
question_types = ['ab', 'compare', 'repeat']

for question_type in question_types:
    with open(f"data/response_templates/{question_type}.json", encoding="utf-8") as f:
        response_patterns[question_type] = json.load(f)


def evaluate(scenario, model, eval_nb_samples, max_tokens, eval_temp, eval_top_p, previous_messages: List[Dict] = None, system_msg: bool = True, system_pre: str = ""):
    results = []
    question_types = ['ab', 'compare', 'repeat']
    for question_type in question_types:
        for question_ordering in [0, 1]:
            # Get question form
            question_form, action_mapping = get_question_form(
                scenario=scenario,
                question_type=question_type,
                question_ordering=question_ordering,
                system_instruction=True,
            )

            # Set result base dict
            result_base = {
                "scenario_id": scenario["scenario_id"],
                "model_id": model.get_model_id(),
                "question_type": question_type,
                "question_ordering": question_ordering,
                "question_header": question_form["question_header"],
                "question_text": question_form["question"],
                "eval_technique": "top_p_sampling",     # can just add at the end?
                "eval_top_p": eval_top_p,               # can just add at the end?
                "eval_temperature": eval_temp,          # can just add at the end?
            }

            for nb_query in range(eval_nb_samples):
                result_base["eval_sample_nb"] = nb_query

                messages = []

                if previous_messages:
                    messages.extend(previous_messages)

                if system_msg:
                    messages.extend([
                        {"role": "system", "content": f"{system_pre}\n{question_form['question_header']}"},
                        {"role": "user", "content": question_form["question"]},
                    ])
                    
                    # Query model
                    response = model.get_top_p_answer(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=eval_temp,
                        top_p=eval_top_p,
                    )
                else:
                    messages.extend([
                        {"role": "user", "content": question_form["question"]},
                    ])
                    response = model.get_top_p_answer(
                        messages=messages,
                        system=question_form["question_header"],
                        max_tokens=max_tokens,
                        temperature=eval_temp,
                        top_p=eval_top_p,
                    )

                # Match response (token sequence) to actions
                response["decision"] = token_to_action_matching(
                    response["answer"],
                    scenario,
                    response_patterns,
                    question_type,
                    action_mapping,
                    refusals,
                )

                # Log Results
                result = {**result_base, **response}
                results.append(result)
    return results