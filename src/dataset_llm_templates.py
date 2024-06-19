from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import os
import pandas as pd


def langchain_templates(X_train_orig, y_train_orig, dataset):
    response_schemas = []

    df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)

    for idx, col in enumerate(list(df_orig.columns)):
        if (
            col == "is_dead"
            or col == "mortCancer"
            or col == "death"
            or col == "death_all"
        ):
            resp = ResponseSchema(
                name=col,
                description=f"label if patient dead or not, {col}",
            )
        elif col == "y":
            resp = ResponseSchema(
                name="target",
                description=f"binary label, {col}",
            )
        elif col == "salary":
            resp = ResponseSchema(
                name="target",
                description=f"label if salary above 50K or not, {col}",
            )
        else:
            resp = ResponseSchema(
                name=col,
                description=f"feature column",
            )
        response_schemas.append(resp)

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    if dataset == "covid":
        print("covid")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible. 

        I will give you real examples first

        Leverage your medical knowledge about covid and brazil generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "cutract":
        print("cutract")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your medical knowledge about prostate cancer in the UK to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

       DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "compas":
        print("compas")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your knowledge about criminal recividsm to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "seer":
        print("seer")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your medical knowledge about prostate cancer in the USA to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "support":
        print("support")
        generator_template = """\
        You are a synthetic data generator. 
        You produce data which mirrors \
        the given examples in structure and distributions \
        but also produces diverse samples

        Use your medical knowledge about hospitalized patients to generate realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES
        """

    if dataset == "maggic":
        print("maggic")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your medical knowledge about heart failure patients to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "adult":
        print("adult")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your knowledge of salary above 50K based on demographic features to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "higgs":
        print("higgs")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your knowledge of particle physics and higgs bosons to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "bio":
        print("bio")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your knowledge of chemical properties of molecules causing a biological response to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    if dataset == "drug":
        print("drug")
        generator_template = """\
        You are a synthetic data generator. 
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produces as diverse samples as possible

        I will give you real examples first

        Leverage your knowledge of features leading to drug usage and consumption to generate 1000 realistic but diverse samples. 

        example data: {data}

        {format_instructions}

        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
        """

    df_orig = df_orig.sample(frac=1).reset_index(drop=True)

    prompt = ChatPromptTemplate.from_template(template=generator_template)

    return prompt, generator_template, format_instructions, df_orig
