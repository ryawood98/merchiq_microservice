import json
import os
import time
from datetime import timedelta
from functools import wraps

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

system = """You are a promotion planner for a market-leading CPG brand. You are tasked with creating a winning promotion plan for the next fiscal year. You need to create a promotion plan that is based on your promotion schedule from the previous year, but with improvements. These improvements can take the form of adjusting the timing of your promotions such that their start dates always either match or front-run any competitors' promotions. They can also take the form of changing the frequency, cadence, or depth of discounting to match the dynamics of your competitor's predicted promotion plan. You will be given your promotion schedule from last year in a JSON-formatted list of promotions, and you will also be given a list of promotions which your main competitor is predicted to run in the future year. It is very important that you must always give a professional explanation of your promotion plan, followed by how the competitor's predicted promotion plan informs it. It should not reference you, and the language should be in the style of documentation rather than conversation. You will be given your promotion schedule from the last year a JSON formatted list of promotions is a variable called HISTORICAL. Here is an example of the format of the historical promotions you will receive:

[
    {
        "promo_start_date": "2024-01-01",
        "promo_length_weeks": 4,
        "display_string": "Spend $25 Get $5",
    },
    {
        "promo_start_date": "2024-05-12",
        "promo_length_weeks": 1,
        "display_string": "20% Off",
    },
    ...
]

You will also be given a list of promotions which your main competitor is predicted to run in the future year in a variable called PREDICTED. The format of these predicted promotions will be identical to the one depicted above for the historical case. Here is an example for clarity:
[
    {
        "promo_start_date": "2025-01-17",
        "promo_length_weeks": 2,
        "display_string": "Buy 1 Get 1 50% Off",
    },
    {
        "promo_start_date": "2025-03-14",
        "promo_length_weeks": 3,
        "display_string": "Spend $25 Get 50% Off",
    },
    {
        "promo_start_date": "2025-03-14",
        "promo_length_weeks": 1,
        "display_string": "Weekly Ad",
    },
    ...
]

Your response should be returned in a RFC8259 compliant JSON response list of promotions, you shall follow this format without deviation whatsoever. Additionally, every promotion in the historical list (not necessary at the same predicted time) must exist in the response unless you have a good reason to remove it. You are allowed however to make modifications to the promotions, but every promotion in the historical list must exist in the response. You are not allowed to make wild changes to the promotions (including large percentage discounts for larger durations of time unless there is historical precedent, large price discounts unless there is historical precedent, and so on). You must not change the length of any promotion by more than two weeks. If you remove any promotions from the previous calendar year, please provide a professional explanation for why you did so. Here is an example of the format of the response:

[
    {
        "promo_start_date": "2025-01-10",
        "promo_length_weeks": 2,
        "display_string": "Buy 1 Get 1 50% Off",
        "explanation": [INSERT REASONING FOR WHY YOU CHOSE TO PLAN A PROMOTION THIS WEEK]
    },
    ...
]

Please use the above directions diligently and carefully. Ultimately, you are the expert here and you should trust your opinion and decision making. Feel free to make modifications to the promotion plan but be mindful and cautious about doing so and you should prioritize front-running competitor's promotions when it makessense and maximizing revenue for your brand.

You will not attach ``` or ```json to the response to signify that the response is a JSON. It is already assumed that is the case.
"""


def retry(num_attempts=5, wait_time=4, wait_multiplier=1.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            extra_prompts = []
            for attempt in range(num_attempts):
                try:
                    return func(*args, **kwargs, extra_prompts=extra_prompts)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    extra_prompts.append(
                        [
                            "Please be reminded that your output must be a valid JSON response. Something that one "
                            "can just directly parse into JSON without any additional processing.",
                        ]
                    )
                    if attempt < num_attempts - 1:  # Don't sleep on the last attempt
                        sleep_time = wait_time * (wait_multiplier**attempt)
                        time.sleep(sleep_time)
            raise last_exception or Exception(f"Failed after {num_attempts} attempts")

        return wrapper

    return decorator


@retry(num_attempts=7, wait_time=2, wait_multiplier=1.5)
def query_openai(system, user, extra_prompts=[]):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for prompt in extra_prompts:
        messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return json.loads(response.to_dict()["choices"][0]["message"]["content"])


def generate_promo_plan(
    promo_pred: pd.DataFrame, promo_prev: pd.DataFrame
) -> list[dict]:

    # parse promo_pred into a list of dicts
    promo_pred["promo_start_date"] = promo_pred["promo_start_date"].apply(
        lambda x: x.isoformat()
    )
    promo_prev["promo_start_date"] = promo_prev["promo_start_date"].apply(
        lambda x: x.isoformat()
    )
    promo_pred_list = json.dumps(
        promo_pred[
            ["promo_start_date", "display_string", "promo_length_weeks", "promo_type"]
        ].to_dict("records")
    )
    promo_prev_list = json.dumps(
        promo_prev[
            ["promo_start_date", "display_string", "promo_length_weeks", "promo_type"]
        ].to_dict("records")
    )

    user = f"""
    HISTORICAL={promo_pred_list}
    PREDICTED={promo_prev_list}
    """

    promo_generated_list = query_openai(system, user)

    # process each promo to make sure values are valid for Next.js app
    for promo in promo_generated_list:
        promo["promo_start_date"] = pd.to_datetime(promo["promo_start_date"])
        if (
            promo["promo_start_date"].weekday() != 6
        ):  # ensure start date is a Sunday, otherwise shift it to the nearest Sunday
            if promo["promo_start_date"].weekday() < 3:
                promo["promo_start_date"] = promo["promo_start_date"] - timedelta(
                    days=promo["promo_start_date"].weekday() + 1
                )
            elif promo["promo_start_date"].weekday() >= 3:
                promo["promo_start_date"] = promo["promo_start_date"] + timedelta(
                    days=6 - promo["promo_start_date"].weekday()
                )
        promo["promo_end_date"] = promo["promo_start_date"] + timedelta(
            days=7 * promo["promo_length_weeks"] - 1
        )
        promo["retailer_week"] = [
            promo["promo_start_date"] + timedelta(days=7 * i)
            for i in range(promo["promo_length_weeks"])
        ]
    return promo_generated_list
