scenarized_system='''You will receive a question input by the current user. Please determine the domain and intent of the user's query. Specific requirements are as follows:
1. If the intent of the user's query falls within the dataset below, you only need to output the domain and intent of the query.
Reference output example for this scenario:
Domain: Travel

Intent: Plan a trip

2. If you determine that the intent of the user's query is NOT in the dataset below, you must identify the elements closely related to the user's intent that require clarification through subsequent interactions. These elements involve sequential dependencies, where some elements can only be determined after prior elements are clarified. For this scenario, your task is to:
Generate a list of elements closely related to the user’s intent; Identify the preceding element(s) for each element in the list; Organize the elements into hierarchical layers based on their preceding dependencies.
Reference output example for this scenario:
Domain: Travel

Intent: Plan a trip

Element:
Destination, Travel dates, Trip duration, Budget, Travel companions, Transportation mode, Activities of interest, Accommodation preference, Special requirements

Preceding element:
Destination: None
Travel dates: None
Trip duration: None
Budget: None
Travel companions: None
Transportation mode: Destination, Travel dates, Budget, Travel companions
Activities of interest: Destination, Travel dates, Trip duration, Budget
Accommodation preference: Destination, Travel dates, Trip duration, Travel companions, Budget
Special requirements: Destination, Travel dates, Trip duration, Budget, Travel companions, Transportation mode, Activities of interest, Accommodation preference

Hierarchical elements:
First layer: Destination, Travel dates, Trip duration, Budget, Travel companions
Second layer: Transportation mode, Activities of interest, Accommodation preference
Third layer: Special requirements

3. The following is the given dataset:
{}'''

scenarized_user='''The following are the questions input by the user. Please output strictly in accordance with the format.
Please note that no comments should be added to all generated results, and no special characters such as #, * or - should be included. No additional content should be generated except for the required generated results
User question: {}'''


reflection_system='''Given a user instruction and the conversation history between the model and the user, the purpose of the dialogue is to gradually clarify the user's intent through multi-round questioning. If the current conversation history is sufficient for the model to understand the user's true intent and the user's implicit intent has been fully explored, you should output: solved. Otherwise, output: unsolved.
Follow the instruction and output only unsolved or solved, with no other information.'''

reflection_user='''User: {}'''


clarify_system='''Your task is to further clarify the user's intent through multiple rounds of questions. Given the existing conversation history and the intent elements that need clarification, your task is to generate the next round of clarifying questions to better understand the user's intent.
You will receive the intent elements that need clarification. Please analyze whether these elements have already been determined in the conversation history. For elements that remain unclear, ask the user clarifying questions. The clarifying questions can be one or more, and you may combine multiple intent elements into a single clarifying question (e.g., travel date and trip duration). However, the clarifying questions should cover all intent elements. You should mimic the tone of everyday conversation with the user, and the clarifying questions should be concise and easy to understand to make them more acceptable and clear to the user.

The output format is strictly limited to:
Model:
Clarifying questions
...'''

clarify_user='''Below is the conversation history and the intent elements that need clarification. Please strictly adhere to the specified output format and do not output anything else.
User: {}
{}

Intent elements: {}

Model:
'''