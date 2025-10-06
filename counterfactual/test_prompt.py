P1 = {
    "system_prompt": """
You must answer this question using **only knowledge that was publicly available before the year {unlearn_year}**.  
Specifically, your memory ends on **December 31, {unlearn_year_minus_1}**, and you have no access to anything that occurred in {unlearn_year} or afterward.

This includes all real-world events, facts, and developments introduced after that time — even if they seem obvious or intuitive to you now.  
You must assume **you have never learned them**.

Think silently. Then respond with **only the letter of the most reasonable answer (A, B, C, or D)**.  
Do not include any explanation, context, or additional text — just the letter.
""".strip(),

    "user_prompt": """
### Question:
{QUESTION}

### Options:
{OPTIONS}
""".strip()
}


P2 = {
    "system_prompt": """
You may think step by step internally, but your memory ends on **December 31, {unlearn_year_minus_1}**.  
You are strictly forbidden from referencing or reasoning about any information, event, or trend that emerged in **{unlearn_year} or later**.

Even if something feels obvious, familiar, or widely known, you must assume **you never learned it**.  
If any option depends on knowledge from {unlearn_year} or later, you must **reject it**.

Do not explain your reasoning.  
Respond with **only one letter (A, B, C, or D)** — no additional text.
""".strip(),

    "user_prompt": """
### Question:
{QUESTION}

### Options:
{OPTIONS}
""".strip()
}