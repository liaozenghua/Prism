from scenario.scenario_utils import *
from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
client = OpenAI(api_key="sk-181fc3c3aa4b4a42a12d275d6c33b5ee", base_url="https://api.deepseek.com")


system_prompt='''对于用户输入的模糊意图，大模型应该有一套意图理解逻辑，需要识别有哪些元素是与输入意图密切相关需要后续通过与用户交互进行明确的。并且这些元素是涉及到先后顺序的，有些元素需要在上一元素明确之后才能确定。
对于输入的意图，你的任务是生成与用户意图密切相关的元素列表，并且为元素列表中的每个元素生成前置元素，最后根据每个元素的前置元素条件，生成每个层次的元素。
请注意所有生成的结果不加入注释，不加入#或者*或者-等特殊字符，除要求生成的结果之外不额外生成任何内容。

以下是参考样例1：
Domain: Travel

Intent: Plan a trip

Element:
Destination,Travel dates,Trip duration,Budget,Travel companions,Transportation mode,Activities of interest,Accommodation preference,Special requirements

Preceding element:
Destination: None
Travel dates: None
Trip duration: None
Budget: None
Travel companions: None
Transportation mode: Destination,Travel dates,Budget,Travel companions
Activities of interest: Destination,Travel dates,Trip duration,Budget
Accommodation preference: Destination,Travel dates,Trip duration,Travel companions,Budget
Special requirements: Destination,Travel dates,Trip duration,Budget,Travel companions,Transportation mode,Activities of interest,Accommodation preference

Hierarchical elements:
First layer: Destination,Travel dates,Trip duration,Budget,Travel companions
Second layer: Transportation mode,Activities of interest,Accommodation preference
Third layer: Special requirements

参考样例2：
Domain: Travel

Intent: Buy travel insurance

Element: Travel dates, Destination, Traveler's age, Trip purpose, Coverage level, Medical conditions, Pre-existing conditions, Deductible preference, Special requirements

Preceding element:
Travel dates: None
Destination: None
Traveler's age: None
Trip purpose: None
Coverage level: Travel dates, Destination, Traveler's age
Medical conditions: Traveler's age
Pre-existing conditions: Coverage level, Medical conditions
Deductible preference: Coverage level
Special requirements: Travel dates, Destination, Traveler's age, Coverage level, Medical conditions, Pre-existing conditions

Hierarchical elements:
First layer: Travel dates, Destination, Traveler's age, Trip purpose
Second layer: Coverage level, Medical conditions
Third layer: Pre-existing conditions, Deductible preference
Fourth layer: Special requests'''

user_prompt='''以下是当前输入的意图，请你严格遵循参考样例的格式输出：
Domain: {}

Intent: {}'''

system_prompt1='''You are a master of data annotation.'''


user_prompt1='''我正在进行数据标注的一项工作，找出每个元素的前置元素，流程是我获取了大模型得出的结果，然后独立请三个标注者进行人工修正，接着三个标注者共通商讨得到标准答案。
现在我给你标准答案，请你反推生成大模型得出的结果，第一个标注者修正的结果，第二个标注者修正的结果，第三个标注者修正的结果。请注意所有生成的结果不加入注释，不加入#或者*或者-等特殊字符，除要求生成的结果之外不额外生成任何内容。
Domain: {}

Intent: {}

标准答案:
{}

大模型得出的结果:

第一个标注者修正的结果:

第二个标注者修正的结果:

第三个标注者修正的结果:'''















data=[]
process_data=[]

all_sheets = pd.read_excel(os.path.abspath(r'/mnt/d/study/code/ReST-SDLG/data/data.xlsx'), sheet_name='Sheet1')


for id,i in enumerate(tqdm(all_sheets['Domain'])):
    if id<=345:
        continue
    domain=all_sheets['Domain'][id]
    intent=all_sheets['Intent'][id]
    example=all_sheets['Example'][id]
    prompt=user_prompt.format(domain,intent)


    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False,
        timeout=300
    )
    answer=response.choices[0].message.content

    # 提取Element部分
    element_start = answer.find("Element: ") + len("Element: ")
    element_end = answer.find("Preceding element:")
    Element = answer[element_start:element_end].strip()

    # 提取Preceding element部分
    preceding_start = answer.find("Preceding element:") + len("Preceding element:")
    preceding_end = answer.find("Hierarchical elements:")
    Preceding_element = answer[preceding_start:preceding_end].strip()

    # 提取Hierarchical elements部分
    hierarchical_start = answer.find("Hierarchical elements:") + len("Hierarchical elements:")
    Hierarchical_elements = answer[hierarchical_start:].strip()

    prompt1=user_prompt1.format(domain,intent,Preceding_element)

    response1 = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt1},
            {"role": "user", "content": prompt1},
        ],
        temperature=1.0,
        stream=False
    )
    answer1 = response1.choices[0].message.content


    result_start = answer1.find("大模型得出的结果:") + len("大模型得出的结果:")
    result_end = answer1.find("第一个标注者修正的结果:")
    result = answer1[result_start:result_end].strip()

    labelled1_start = answer1.find("第一个标注者修正的结果:") + len("第一个标注者修正的结果:")
    labelled1_end = answer1.find("第二个标注者修正的结果:")
    labelled1 = answer1[labelled1_start:labelled1_end].strip()

    labelled2_start = answer1.find("第二个标注者修正的结果:") + len("第二个标注者修正的结果:")
    labelled2_end = answer1.find("第三个标注者修正的结果:")
    labelled2 = answer1[labelled2_start:labelled2_end].strip()

    labelled3_start = answer1.find("第三个标注者修正的结果:") + len("第三个标注者修正的结果:")
    labelled3 = answer1[labelled3_start:].strip()

    data.append({'domain':domain,'intent':intent,"element":Element,'example':example,'preceding_element':Preceding_element,'hierarchical_elements':Hierarchical_elements,'result':result,'labelled1':labelled1,'labelled2':labelled2,'labelled3':labelled3})
    dump_json(data, '/mnt/d/study/code/ReST-SDLG/scenario/data_346_{}.json'.format(id), indent=4)

    Element_process=Element.replace(', ',',').split(',')
    example_process=example.split('\n')

    a=Preceding_element.split('\n')
    Preceding_element_process={}
    for jd,j in enumerate(a):
        b=j.strip().split(': ')
        b[1]=b[1].replace(', ',',').split(',')
        if b[1]==['None']:
            Preceding_element_process[b[0]] = None
        else:
            Preceding_element_process[b[0]]=b[1]

    a = Hierarchical_elements.split('\n')
    Hierarchical_elements_process = []
    for jd, j in enumerate(a):
        b = j.split('layer: ')[-1].strip()
        b=b.replace(', ',',').split(',')
        Hierarchical_elements_process.append(b)

    process_data.append({'domain':domain,'intent':intent,"element_process":Element_process,'example_process':example_process,'preceding_element_process':Preceding_element_process,'hierarchical_elements_process':Hierarchical_elements_process})

    dump_json(process_data, '/mnt/d/study/code/ReST-SDLG/scenario/data_process_346_{}.json'.format(id), indent=4)
print(1)
