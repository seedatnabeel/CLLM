import re
import json
import pandas as pd
from copy import deepcopy
import os
import openai
from langchain.prompts import ChatPromptTemplate


def llm_gen(
    prompt,
    generator_template,
    format_instructions,
    example_df,
    llm_serving,
    api_details,
    n_samples=100,
    temperature=0.75,
    max_tokens=8000,
    model="gpt4_20230815",
    n_processes=10,
    ic_samples=20,
):

    init = True
    not_sufficient = True
    df_list = []
    for i in range(500):
        # shuffle pandas dataframe rows
        try:
            example_df = example_df.sample(n=ic_samples, replace=True).reset_index(
                drop=True
            )

            small_data = str(example_df.to_dict(orient="records"))

            prompt = ChatPromptTemplate.from_template(template=generator_template)

            messages = prompt.format_messages(
                data=small_data, format_instructions=format_instructions
            )

            if llm_serving == "together":
                openai.api_base = api_details["api_base"]
                openai.api_key = api_details["api_key"]

            if llm_serving == "vllm":
                from openai import OpenAI

                # Set OpenAI's API key and API base to use vLLM's API server.
                openai_api_key = "EMPTY"
                openai_api_base = "http://localhost:8000/v1"

            if llm_serving == "azure_openai":
                openai.api_type = "azure"
                openai.api_base = api_details["api_base"]
                openai.api_version = api_details["api_version"]
                openai.api_key = api_details["api_key"]

            if llm_serving != "vllm":
                messages = [
                    {
                        "role": "system",
                        "content": "You are a tabular synthetic data generation model.",
                    },
                    {"role": "user", "content": messages[0].content},
                ]

            else:
                prompt = messages[0].content
                prompt = "".join(messages[0].content.split("\n")[1:])
                messages = [
                    {
                        "role": "system",
                        "content": "You are a synthetic data generator.",
                    },
                    {"role": "user", "content": f"{prompt}"},
                ]

            if llm_serving == "azure_openai":
                response = openai.ChatCompletion.create(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    n=n_processes,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )

            if llm_serving == "together":
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    n=n_processes,
                )

            if llm_serving == "vllm":
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.95,
                    n=n_processes,
                    frequency_penalty=0,
                    presence_penalty=0,
                    max_tokens=max_tokens,
                    stop=None,
                )

            for idx in range(n_processes):

                try:

                    if llm_serving == "vllm":
                        data = response.choices[idx].message.content
                    else:
                        data = response["choices"][idx]["message"]["content"]

                    # Extract dict-like strings using regular expressions
                    dict_strings = re.findall(r"\{[^{}]*\}", data)

                    # Convert dict-like strings to actual dictionaries
                    dicts = [json.loads(ds) for ds in dict_strings]

                except:
                    continue

                if llm_serving == "vllm":
                    df_tmp = deepcopy(pd.DataFrame(dicts))
                    df_tmp = df_tmp[
                        ~df_tmp.apply(
                            lambda row: any(
                                [
                                    isinstance(cell, str)
                                    and cell
                                    in ["integer", "float", "numeric", "categorical"]
                                    for cell in row
                                ]
                            ),
                            axis=1,
                        )
                    ]
                    df_list.append(df_tmp)

                else:
                    if idx == 0:
                        df_tmp = deepcopy(pd.DataFrame(dicts))
                        df_tmp = df_tmp[
                            ~df_tmp.apply(
                                lambda row: any(
                                    [
                                        isinstance(cell, str)
                                        and cell
                                        in [
                                            "integer",
                                            "float",
                                            "numeric",
                                            "categorical",
                                        ]
                                        for cell in row
                                    ]
                                ),
                                axis=1,
                            )
                        ]

                    else:
                        df_check = pd.DataFrame(dicts)
                        df_check = df_check[
                            ~df_check.apply(
                                lambda row: any(
                                    [
                                        isinstance(cell, str)
                                        and cell
                                        in [
                                            "integer",
                                            "float",
                                            "numeric",
                                            "categorical",
                                        ]
                                        for cell in row
                                    ]
                                ),
                                axis=1,
                            )
                        ]
                        df_tmp = df_tmp.append(df_check, ignore_index=True)

            if llm_serving == "vllm":
                df_tmp = df_list[0]
                for df_check in df_list[1:]:
                    df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)

            else:
                # filter any metadata out
                df_list.append(df_tmp)

            if init == True:
                df_llm = deepcopy(df_tmp)
                init = False
            else:
                if llm_serving == "vllm":
                    df_llm = pd.concat([df_llm, df_tmp], ignore_index=True)
                else:
                    df_llm = df_llm.append(df_tmp, ignore_index=True)

            n_gen = df_llm.shape[0]
            print("Current = ", n_gen, df_llm.shape)

            if n_gen >= n_samples:
                print("Done...")
                print(n_gen, df_llm.shape)
                not_sufficient = False
                break

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(e)
            continue

    return df_llm
