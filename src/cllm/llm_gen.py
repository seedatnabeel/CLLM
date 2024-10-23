import re
import json
import pandas as pd
from copy import deepcopy
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
    
    """
    The function `llm_gen` generates synthetic data based on a given prompt using different language
    model APIs.
    
    Args:
      prompt: The `llm_gen` function you provided seems to be a data generation function that uses a
    language model to generate synthetic data based on a given prompt and example data. Here's an
    explanation of the parameters used in the function:
      generator_template: The `llm_gen` function you provided seems to be a data generation function
    that uses a language model to generate synthetic data based on a given prompt and example data. It
    interacts with different types of language model serving environments like "together", "vllm", and
    "azure_openai"
      format_instructions: The `format_instructions` parameter likely contains instructions on how to
    format the data for the model. It could include details on how to structure the input data, such as
    column names, data types, and any specific formatting requirements the model expects. This
    information is crucial for preparing the input data correctly before feeding
      example_df: The `example_df` parameter in the `llm_gen` function is a pandas DataFrame that
    contains the example data used for generating synthetic data. It is sampled within the function to
    create a smaller dataset (`small_data`) that is then used as input for the language model (LLM) to
    generate
      llm_serving: The `llm_serving` parameter in the function `llm_gen` determines the type of language
    model serving platform that will be used for generating synthetic data. The possible values for
    `llm_serving` in this function are:
      api_details: The `api_details` parameter in the `llm_gen` function likely contains details needed
    for API authentication and endpoint configuration. It may include information such as the API base
    URL, API key, API version, and API type specific to the service being used (e.g., Azure OpenAI).
      n_samples: The `n_samples` parameter in the `llm_gen` function represents the number of synthetic
    data samples that you want to generate. This parameter specifies the total number of data samples
    that the function should aim to generate before stopping and returning the generated synthetic data.
    Defaults to 100
      temperature: The `temperature` parameter in the `llm_gen` function controls the randomness of the
    generated text. A lower temperature value (close to 0) will result in more deterministic outputs,
    where the model is more likely to choose high-probability tokens. On the other hand, a higher
    temperature
      max_tokens: The `max_tokens` parameter in the `llm_gen` function specifies the maximum number of
    tokens (words or subwords) that the language model can generate in response to the input prompt.
    This parameter helps control the length and complexity of the generated text. In the provided
    function, the `max. Defaults to 8000
      model: The `model` parameter in the `llm_gen` function refers to the specific language model that
    will be used for generating synthetic data. In this case, the default value for the `model`
    parameter is set to "gpt4_20230815". This model will be utilized by the. Defaults to gpt4_20230815
      n_processes: The `n_processes` parameter in the `llm_gen` function specifies the number of
    parallel processes to use for generating responses from the language model. In this function, it is
    used to control how many parallel processes are created to interact with the language model for
    generating synthetic data based on the provided inputs. Defaults to 10
      ic_samples: The `ic_samples` parameter in the `llm_gen` function represents the number of samples
    to be taken from the example dataframe for each iteration of the data generation process. In the
    provided code snippet, `ic_samples` is used to sample a subset of the example dataframe before
    formatting the data and. Defaults to 20
    
    Returns:
      The function `llm_gen` returns a pandas DataFrame `df_llm` containing the generated synthetic data
    based on the provided inputs and parameters.
    """

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
                    print(f"idx: {idx}")
                    if idx == 0:
                        df_tmp = deepcopy(pd.DataFrame(dicts))
                        print(f"df_tmp {df_tmp.head()}")
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
                        try:
                            df_tmp = df_tmp.append(df_check, ignore_index=True)
                        except UnboundLocalError as e:
                            print(f"Error occurred: idx - {idx}, llm_serving - {llm_serving}")
                            print(f"df_tmp not previously created should be: {pd.DataFrame(dicts)}")
                            df_tmp = deepcopy(pd.DataFrame(dicts))
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
