P1 = """
You must provide a definition using **only knowledge that was publicly available before the year {unlearn_year}**.
Specifically, your memory ends on **December 31, {unlearn_year_minus_1}**, and you have no access to anything that occurred in {unlearn_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.  
You must assume **you have never learned them**.

Think silently. Then respond with **only a concise, factual definition** of the given word or phrase, based on what you knew **before** {unlearn_year}.

Do **not** include any explanation, context, or additional text — just the definition.
""".strip()

P2 = """
You may think step by step internally, but your memory ends on **December 31, {unlearn_year_minus_1}**.  
You are strictly forbidden from referencing or reasoning about any information, event, or trend that emerged in **{unlearn_year} or later**.

Even if something feels obvious, familiar, or widely known, you must assume **you never learned it**.

Please provide a clear and concise definition of the given word or phrase, as it was understood prior to {unlearn_year}.

Do not explain your reasoning.  
Respond with **only a concise, factual definition** — no additional text.
""".strip()

word_prompt = """
What is the dictionary definition of the word or phrase "{input}"?

Respond concisely with a definition, without examples or commentary.

Your answer must be a single sentence, no longer than 100 words.
""".strip()