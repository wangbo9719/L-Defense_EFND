from openai import OpenAI

OPENAI_KEY = "YOUR_KEY"
OPENAI_BASE = "YOUR_URL"
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE)

EXPLANATION_SYSTEM_PROMPT_v0 = f"""\
You have been specially designed to perform abductive reasoning for the fake news detection task. 
Your primary function is that, according to a veracity label about a news claim and some sentences related to the claim, \
please provide a streamlined rationale, without explicitly indicating the label, for how it is reasoned as the given veracity label. \
Note that the related sentences that may be helpful for the explanation, but they are mixed with noise. 
Thus, the rationale you provided may not necessarily need to rely entirely on the sentences above, \
and there is no need to explicitly mention which sentence was referenced in your explanation. \
Your goal is to output a streamlined rationale that allows people \
to determine the veracity of the claim when they read it, \
without requiring any additional background knowledge.
The length of your explanation should be less than 200 words. 
"""

EXPLANATION_USER_PROMPT_v0 = f"""\
Given a claim: [CLAIM], a veracity label [LABEL], \
please give me a streamlined rationale associate with the claim, without explicitly indicating the label, for how it is reasoned as [LABEL].\
Below are some sentences that may be helpful for the explanation, but they are mixed with noise: 
[SENTENCES]
Note, please do not repeat the claim and the label in your explanation, just directly output your streamlined rationale.
"""

