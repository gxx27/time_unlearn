import os
import json
from tqdm import tqdm
from together import Together
import openai
from test_prompt import *

together_client = Together(api_key="your_api_key")
openai_client = openai.OpenAI(api_key="your_api_key")

prompt_dir = "prompt"
dataset_dir = "semantic.json"
output_dir = "results_before_unlearn"
os.makedirs(output_dir, exist_ok=True)

model_list = {
    "deepseek-ai/DeepSeek-V3": "together",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "together",
    "gpt-4o-2024-08-06": "openai"
}

if __name__ == "__main__":
    with open(dataset_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    for pair in tqdm(data):
        word = pair["Word"]
        year = int(pair["Cutoff year"])
        
        result = {
            "Word": word,
            "Meaning before unlearn": pair["Meaning before unlearn"],
            "Meaning after unlearn": pair["Meaning after unlearn"],
            "model_outputs": {}
        }
        prompt = word_prompt.replace("{input}", word)

        for model, api_type in model_list.items():
            try:
                if api_type == "together":
                    response = together_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0
                    )
                    output_text = response.choices[0].message.content.strip()

                elif api_type == "openai":
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0
                    )
                    output_text = response.choices[0].message.content.strip()

                else:
                    raise ValueError(f"Unknown API type: {api_type}")

                result["model_outputs"][model] = output_text

            except Exception as e:
                print(f"[ERROR] {model} failed on {word}: {e}")
                result["model_outputs"][model] = f"[ERROR] {e}"

        output_file = os.path.join(output_dir, f"{word}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)