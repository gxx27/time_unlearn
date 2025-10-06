import os
import re
import json
from tqdm import tqdm
from test_prompt import *
from together import Together
import openai

together_client = Together(api_key="your_api_key")
openai_client = openai.OpenAI(api_key="your_api_key")

dataset_path = "counterfactual.json"
output_path = "results"
os.makedirs(output_path, exist_ok=True)

model_list = {
    "deepseek-ai/DeepSeek-V3": "together",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "together",
    "gpt-4o-2024-08-06": "openai"
}

prompt_variants = {
    "P1": P1,
    "P2": P2,
}

def extract_answer(text):
    text = text.strip().upper()
    if re.fullmatch(r"[ABCD]", text):
        return text
    return "N/A"

if __name__ == "__main__":
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    file_results = {model: {pstyle: [] for pstyle in prompt_variants} for model in model_list}

    for model, api_type in model_list.items():
        for prompt_style, prompt_template in prompt_variants.items():
            print(f"[{model}] - {prompt_style}")
            for pair in tqdm(data):
                year = int(pair["Cutoff year"])
                try:
                    system_prompt = prompt_template["system_prompt"].format(
                        unlearn_year=year,
                        unlearn_year_minus_1=year - 1
                    )
                    user_prompt = prompt_template["user_prompt"].format(
                        QUESTION=pair["Question"],
                        OPTIONS=pair["Options"]
                    )

                    if api_type == "together":
                        response = together_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=256,
                            temperature=0
                        )
                        output_text = response.choices[0].message.content.strip()

                    elif api_type == "openai":
                        response = openai_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=256,
                            temperature=0
                        )
                        output_text = response.choices[0].message.content.strip()

                    else:
                        raise ValueError(f"Unknown API type: {api_type}")

                    file_results[model][prompt_style].append({
                        "question": pair["Question"],
                        "options": pair["Options"],
                        "model_output": output_text,
                        "extracted_answer": extract_answer(output_text),
                        "answer_before_unlearn": pair.get("Answer before unlearn", "N/A"),
                        "answer_after_unlearn": pair.get("Answer after unlearn", "N/A")
                    })

                except Exception as e:
                    print(f"[ERROR] {model} - {prompt_style} : {e}")

        with open(os.path.join(output_path, "counterfactual.json"), "w", encoding="utf-8") as f:
            json.dump(file_results, f, ensure_ascii=False, indent=2)