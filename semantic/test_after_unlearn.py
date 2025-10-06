import os
import json
import re
from tqdm import tqdm
from test_prompt import *
from together import Together
import openai

together_client = Together(api_key="your_api_key")
openai_client = openai.OpenAI(api_key="your_api_key")

prompt_dir = "prompt"
dataset_dir = "semantic.json"
output_dir = "results_after_unlearn"
os.makedirs(output_dir, exist_ok=True)

model_list = {
    "deepseek-ai/DeepSeek-V3": "together",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "together",
    "gpt-4o-2024-08-06": "openai"
}

prompt_variants = {
    "P1": P1,
    "P2": P2,
}

if __name__ == "__main__":
    with open(dataset_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    for pair in tqdm(data):
        word = pair["Word"]
        year = int(pair["Cutoff year"])
        
        prompt = word_prompt.replace("{input}", word)
        file_results = {model: {pstyle: None for pstyle in prompt_variants} for model in model_list}

        for model, api_type in model_list.items():
            for pstyle, pconfig in prompt_variants.items():
                try:
                    system_prompt = pconfig.format(
                        unlearn_year=year,
                        unlearn_year_minus_1=year - 1
                    )

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]

                    if api_type == "together":
                        response = together_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=256,
                            temperature=0
                        )
                        output_text = response.choices[0].message.content.strip()

                    elif api_type == "openai":
                        response = openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=256,
                            temperature=0
                        )
                        output_text = response.choices[0].message.content.strip()

                    else:
                        raise ValueError(f"Unknown API type: {api_type}")

                    file_results[model][pstyle] = output_text

                except Exception as e:
                    print(f"[ERROR] {model} - {pstyle} - {word}: {e}")
                    file_results[model][pstyle] = f"[ERROR] {e}"

        final_result = {
            "Word": word,
            "Meaning before unlearn": pair["Meaning before unlearn"],
            "Meaning after unlearn": pair["Meaning after unlearn"],
            "model_outputs": file_results
        }

        output_file = os.path.join(output_dir, f"{word}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)