
SYS_MESSAGE = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

SYS_MESSAGE_TQA = "Provide a brief and concise answer to the question."

SYS_MESSAGE_MATH = """Solve the following math problem efficiently and clearly:

    - For simple problems (2 steps or fewer):
    Provide a concise solution with minimal explanation.

    - For complex problems (3 steps or more):
    Use this step-by-step format:

    ## Step 1: [Concise description]
    [Brief explanation and calculations]

    ## Step 2: [Concise description]
    [Brief explanation and calculations]

    ...

    Regardless of the approach, always conclude with:

    Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

    Where [answer] is just the final number or expression that solves the problem."""


REVISE_PROMPT_NoQ = "Rewrite the text below to fix repetitive language and fill in any unfinished sentences, maintaining the original intent:\n{}" #\nDirectly return the revised answer:

REVISE_PROMPT_ICL ="""Given a question <Q> and its corresponding answer <A>, rewrite the answer under <R> to improve clarity by:
1. Preserving the original meaning and intent.
2. Eliminating repetitive language.
3. Completing any unfinished sentences.
4. Only revising <A> without generating a new question or introducing multi-turn dialogue.


<Q>What are the three primary colors?
<A>The three primary colors are red, blue, and yellow and yellow and yellow. These colors are called primary These colors are called primary These colors are called primary 
<R>The three primary colors are red, blue, and yellow. These colors are called primary because they cannot be created by mixing other colors. 

<Q>Give three tips for staying healthy.
<A>1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole 
<R>1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases. 2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week. 3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.

<Q>{}
<A>{}
<R>"""

REVISE_PROMPT ="""Given a question <Q> and its corresponding answer <A>, rewrite the answer under <R> to improve clarity by:
1. Improve the fluency of the answer without reducing its length.
2. Eliminating repetitive language.
3. Completing any unfinished sentences.
4. Preserving the original meaning and intent.

<Q>{}
<A>{}
<R>"""

REVISE_PROMPT_SYS = """Given a question and its corresponding answer, rewrite the answer to improve clarity by:
1. Preserving the original meaning and intent.
2. Eliminating repetitive language.
3. Completing any unfinished sentences.
4. Only revising answer without generating a new question or introducing multi-turn dialogue.
"""


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
RM_LLAMA_2_7B_OASST1_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}{{ ' prompter: ' + message['content']}}{% elif message['role'] == 'assistant' %}{{ ' assistant: '  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ ' assistant: ' }}{% endif %}{% endfor %}"
RM_GPT_2_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}{{ '\n\nHuman: ' + message['content']}}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: '  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}{% endfor %}"
RM_MISTRAL_7B_DPA_TEMPLATE = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
RM_ULTRARM_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}{{ '\n\nHuman: ' + message['content']}}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: '  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}{% endfor %}"
PYTHIA_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}{{ '<human>' + message['content']}}{% elif message['role'] == 'assistant' %}{{ '<bot>'  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<bot>' }}{% endif %}{% endfor %}"
# PYTHIA_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}{{ 'Human: ' + message['content']}}{% elif message['role'] == 'assistant' %}{{ ' Assistant: '  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ ' Assistant: ' }}{% endif %}{% endfor %}"