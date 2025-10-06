import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import pandas as pd

before_dir = "results_before_unlearn"
after_dir = "results_after_unlearn"

model_names = [
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "gpt-4o-2024-08-06"
]
prompt_styles = ["P1", "P2"]

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
records = []

for filename in tqdm(sorted(os.listdir(before_dir))):
    if not filename.endswith(".json"):
        continue

    before_path = os.path.join(before_dir, filename)
    after_path = os.path.join(after_dir, filename)
    if not os.path.exists(after_path):
        continue

    with open(before_path, "r", encoding="utf-8") as f:
        before_data = json.load(f)
    with open(after_path, "r", encoding="utf-8") as f:
        after_data = json.load(f)

    word = before_data["Word"]
    dict_def = before_data["Meaning before unlearn"]
    golden_def = after_data["Meaning after unlearn"]
    dict_emb = embed_model.encode(dict_def, convert_to_tensor=True)
    golden_emb = embed_model.encode(golden_def, convert_to_tensor=True)

    for model in model_names:
        if model not in before_data["model_outputs"]:
            continue
        if model not in after_data["model_outputs"]:
            continue

        before_output = before_data["model_outputs"][model]
        before_emb = embed_model.encode(before_output, convert_to_tensor=True)
        sim_before_1 = float(util.cos_sim(dict_emb, before_emb))
        sim_before_2 = float(util.cos_sim(golden_emb, before_emb))

        for pstyle in prompt_styles:
            if pstyle not in after_data["model_outputs"][model]:
                continue

            after_output = after_data["model_outputs"][model][pstyle]
            after_emb = embed_model.encode(after_output, convert_to_tensor=True)
            sim_after_1 = float(util.cos_sim(dict_emb, after_emb))
            sim_after_2 = float(util.cos_sim(golden_emb, after_emb))

            records.append({
                "word": word,
                "model": model,
                "prompt_style": pstyle,
                "similarity_before_1": sim_before_1,
                "similarity_before_2": sim_before_2,
                "similarity_after_1": sim_after_1,
                "similarity_after_2": sim_after_2
            })

df = pd.DataFrame(records)
df.to_csv("similarity_records.csv", index=False, encoding="utf-8-sig")
