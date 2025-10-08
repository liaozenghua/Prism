from scenario.scenario_utils import *
from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
client = OpenAI(api_key="sk-181fc3c3aa4b4a42a12d275d6c33b5ee", base_url="https://api.deepseek.com")


system_prompt='''The first column in the table represents the field to which the user's questions to the large model belong, and the second column represents the user's intention of the questions. Your task is to generate five possible questions from users for each intention. The questions should be as complete as possible and not include specific names of people, locations, times, dates or other entities that may cause interference. These entities should be represented by placeholders, such as [location], so as not to interfere with the subsequent similarity calculation. The main purpose is to express the sentence structure in which the user might ask questions. It can be a question or an affirmative sentence, and the length can be long or short. You need to imitate the user's questions.

Please follow the output format of the following case and do not produce any other content.

Domain: Travel
Intent: Plan a trip
Question: 
I want to plan a multi-day trip to [region/country], what's a good itinerary?
Help me plan a relaxing vacation for [number] people focused on [activity].
Looking for trip ideas combining [city/area] and [nearby city/area] within [timeframe].
How should I structure a trip to [destination] to see both major sights and hidden gems?
Need a detailed travel plan for visiting [country] during [season].'''

user_prompt='''Here are the current input domain and intent. Please generate the user's question:

Domain: {}
Intent: {}
Question:
'''


data=[]
process_data=[]

all_sheets = pd.read_excel(os.path.abspath(r'/home/data/liaozenghua/ReST-SDLG/scenario/example.xlsx'), sheet_name='Sheet1')


examples=[]
for id,i in enumerate(tqdm(all_sheets['domain'])):

    domain=all_sheets['domain'][id]
    intent=all_sheets['intent'][id]
    prompt=user_prompt.format(domain,intent)


    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        stream=False,
        timeout=300
    )
    answer=response.choices[0].message.content

    example=answer.split('\n')
    examples.append(example)
    dump_json(examples, '/home/data/liaozenghua/ReST-SDLG/scenario/example_{}.json'.format(id), indent=4)
print(1)
