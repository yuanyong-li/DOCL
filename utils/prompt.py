from langchain.prompts import PromptTemplate
from textwrap import dedent

# 11.11 22:00
rag_prompt_template = PromptTemplate(
    input_variables=["reference_examples", "input_example"],
    template=dedent("""
    ## Problem Statement

    Hate speech refers to statements or expressions that disparage, intimidate, or incite prejudice and hostility against individuals or groups based on characteristics such as race, ethnicity, nationality, religion, sexual orientation, gender identity, disability, or other similar traits.

    Given an Input Post, please analyze it thoroughly and determine whether it constitutes hate speech. To make this decision, follow these steps:

    1. **Select the Most Similar Example**: From the Reference Examples provided, choose the example most similar to the Input Post based on topic, target group, and tone of sentiment expressed. This selected example should serve as a primary guide for classification. Consider the label assigned to the selected example as a reference point when deciding the classification label for the Input Post.

    2. **Compare and Classify**: Using the selected example as a guide, provide:
       - **Explanation**: Justify whether the Input Post contains hate speech, making comparisons with the selected example and considering its label as part of your reasoning.
       - **Classification**: Assign the label based on your analysis:
         - **explicit_hate**: The post contains clear and overt expressions of hate speech with explicit language that conveys strong hostility or aggression toward a group or individual.
         - **implicit_hate**: The post contains subtle or indirect hate speech, suggesting prejudice or offense toward a group or individual without using explicit hostile language.
         - **not_hate**: The post does not contain any form of hate speech or prejudice.

    ## Reference Examples
    : {reference_examples}

    ## Input Post
    : {input_example}

    ## Expected Output:
    Please return the output in the following JSON format:
    {{
        "selected example": "the example most similar to the Input Post"
        "explain": "Your explanation here",
        "result": "explicit_hate or implicit_hate or not_hate"
    }}
        """)
)


docl_prompt_template = PromptTemplate(
    input_variables=["reference_examples", "input_example"],
    template=dedent("""
    ## Problem Statement

    Hate speech refers to statements or expressions that disparage, intimidate, or incite prejudice and hostility against individuals or groups based on characteristics such as race, ethnicity, nationality, religion, sexual orientation, gender identity, disability, or other similar traits.

    Given an `Input Post`, please analyze it thoroughly and determine whether it constitutes hate speech. To make this decision, follow these steps:

    1. **Select Similar Examples**: From the `Reference Examples` provided, select the examples most similar to the `Input Post`. Focus on those that best match in tone, language, and intent.
    2. **Compare and Classify**: Using the selected examples as a guide, provide:
       - **Explanation**: Justify whether the `Input Post` contains hate speech, drawing comparisons to the selected examples.
       - **Classification**: Assign the label:
         - **explicit_hate**: The post contains clear and overt expressions of hate speech with explicit language that conveys strong hostility or aggression toward a group or individual.
         - **implicit_hate**: The post contains subtle or indirect hate speech, suggesting prejudice or offense toward a group or individual without using explicit hostile language.
         - **not_hate**: The post does not contain any form of hate speech or prejudice.

    ## Reference Examples

    {reference_examples}

    ## Task

    Analyze the following `Input Post` and complete these steps:

    1. **Similar Examples**: List the reference examples most similar to the `Input Post`.
    2. **Explanation**: Explain why you classified it as `explicit_hate`, `implicit_hate`, or `not_hate`, referencing the similar examples.
    3. **Classification Label**: Choose one of the following labels:
       - explicit_hate
       - implicit_hate
       - not_hate

    ```Input Post```: "{input_example}"

    """)
)




zs_prompt_template = PromptTemplate(
    input_variables=["input_example"],
    template=dedent("""
    Determine whether the following post is hate speech. 
    Post: {input_example}
    Options: 
        Hate
        Not hate
    Answer:

    Expected Output:
    Please return the output in the following JSON format:
    {{
        "result": "Hate or Not Hate"
    }}
    """)
)


fs_prompt_template = PromptTemplate(
    template=dedent("""
    Determine whether the following post is hate speech. 

    Example 1:
    Post: "non-white candidate for london mayor : there are too many white men | read more :"
    Options: 
        Hate
        Not hate
    Answer: Hate

    Example 2:
    Post: "i hate isis"
    Options: 
        Hate
        Not hate
    Answer: Not hate

    Example 3:
    Post: "human being is a white termanimals ... are what is left when whites close their lips"
    Options: 
        Hate
        Not hate
    Answer: Hate

    Example 4:
    Post: "socialism is retarded . i will give you credit for not promoting race hate though ."
    Options: 
        Hate
        Not hate
    Answer: Not hate

    Post: {}
    Options: 
        Hate
        Not hate
    Answer:

    Expected Output:
    Please return the output in the following JSON format:
    {{
        "result": "Hate or Not Hate"
    }}
    """)
)