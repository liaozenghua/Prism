import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pandas as pd
from tqdm import tqdm
from models_GPT.model import gpt,deepseek
from MCTS_GPT.task import MCTS_Task
from MCTS_GPT.mcts_utils import read_jsonl, read_json, save_pickle
from openai import OpenAI
from MCTS_GPT.prompts import *


def excel_to_text(df, format="markdown"):
    """Convert DataFrame to specified text format."""
    if format == "csv":
        return df.to_csv(index=False)
    elif format == "markdown":
        return df.to_markdown(index=False)
    else:
        return df.to_string(index=False)  # Default to plain text


def extract_scenario(response, df,rag):
    """Extract scenario information from API response."""
    domain = response[response.find("Domain:") + len("Domain:"):response.find("Intent:")].strip()

    if response.find("Element:") != -1:
        intent = response[response.find("Intent:") + len("Intent:"):response.find("Element:")].strip()
    else:
        intent = response[response.find("Intent:") + len("Intent:"):].strip()

    if intent in df['intent'].tolist():
        index = df['intent'].tolist().index(intent)
        scenario = rag[index]
    else:
        element = response[
                  response.find("Element:") + len("Element:"):response.find("Preceding element:")].strip().replace(', ',
                                                                                                                   ',').split(
            ',')
        preceding_element = response[response.find("Preceding element:") + len("Preceding element:"):response.find(
            "Hierarchical elements:")].strip()

        preceding_dict = {}
        for j in preceding_element.split('\n'):
            key, values = j.strip().split(': ')
            values = values.replace(', ', ',').split(',')
            preceding_dict[key] = None if values == ['None'] else values

        hierarchical_elements = response[
                                response.find("Hierarchical elements:") + len("Hierarchical elements:"):].strip()
        hierarchical_list = []
        for j in hierarchical_elements.split('\n'):
            elements = j.split('layer: ')[-1].strip().replace(', ', ',').split(',')
            hierarchical_list.append(elements)

        scenario = {
            'domain': domain,
            'intent': intent,
            "element": element,
            'preceding_element': preceding_dict,
            'hierarchical_elements': hierarchical_list
        }

    return scenario


def run_pipeline(
        data_path,
        rag_path,
        excel_path,
        output_path,
        model,
        model_type
):
    # Load data
    data = read_jsonl(data_path)
    rag = read_json(rag_path)

    # Process Excel file
    df = pd.read_excel(excel_path)
    text_data = excel_to_text(df, format="markdown")

    results = []
    for item in tqdm(data, desc="Processing items"):
        question = item['task']

        if model_type=='deepseek':
            # Get deepseek response
            response = deepseek(
                scenarized_system.format(text_data),
                scenarized_user.format(question),
                model,
                temperature=0.0
            ).choices[0].message.content
        elif model_type=='gpt':
            response = gpt(
                scenarized_system.format(text_data),
                scenarized_user.format(question),
                model,
                temperature=0.0
            ).choices[0].message.content

        # Extract scenario information
        scenario = extract_scenario(response, df,rag)

        # Run MCTS task
        task = MCTS_Task(
            question,
            scenario,
            model_type,
            model,
            'local',
            lang='en'
        )
        final_answer, root = task.run()

        # Update results
        item['final_answer'] = final_answer
        item['root'] = root
        results.append(item)

        # Periodically save results
        save_pickle(results, output_path)
        torch.cuda.empty_cache()

    print("Processing completed successfully")


if __name__ == "__main__":
    # Configuration parameters
    config = {
        "data_path": "/home/data/liaozenghua/ReST-SDLG/data/IN3/train.jsonl",
        "rag_path": "/home/data/liaozenghua/ReST-SDLG/scenario/rag.json",
        "excel_path": "/home/data/liaozenghua/ReST-SDLG/scenario/scenario.xlsx",
        "output_path": "/home/data/liaozenghua/ReST-SDLG/MCTS/mcts_results/in3_results.json",
        "model": "gpt-4-turbo",
    }
    if 'deepseek' in config['model']:
        config['model_type']='deepseek'
    elif 'gpt' in config['model']:
        config['model_type']='gpt'

    run_pipeline(**config)

