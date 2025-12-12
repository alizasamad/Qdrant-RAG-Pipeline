from langchain_core.prompts import PromptTemplate

# Create a prompt template to generate a question a end-user could have about a given context
INITIAL_QUESTION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template="""
    <Instructions>
    Here is some context:
    <context>
    {context}
    </context>

    Your task is to generate 1 question that can be answered using the provided context, following these rules:

    <rules>
    1. The question should make sense to humans even when read without the given context.
    2. The question should be fully answered from the given context.
    3. The question should be framed from a part of context that contains important information. It can also be from tables, code, etc.
    4. The answer to the question should not contain any links.
    5. The question should be of moderate difficulty.
    6. The question must be reasonable and must be understood and responded by humans.
    7. Do not use phrases like 'provided context', etc. in the question.
    8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
    9. The question should not contain more than 10 words, make use of abbreviations wherever possible.
    </rules>

    To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above.

    Output only the generated question with a "?" at the end, no other text or characters.
    </Instructions>
    
    """)

ANSWER_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    <Instructions>
    <Task>
    <role>You are an experienced QA Engineer for building large language model applications.</role>
    <task>It is your task to generate an answer to the following question <question>{question}</question> only based on the <context>{context}</context></task>
    The output should be only the answer generated from the context.

    <rules>
    1. Only use the given context as a source for generating the answer.
    2. Be as precise as possible with answering the question.
    3. Be concise in answering the question and only answer the question at hand rather than adding extra information.
    </rules>

    Only output the generated answer as a sentence. No extra characters.
    </Task>
    </Instructions>
    
    Assistant:""")

SOURCE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""Human:
    <Instructions>
    Here is the context:
    <context>
    {context}
    </context>

    Your task is to extract the relevant sentences from the given context that can potentially help answer the following question. You are not allowed to make any changes to the sentences from the context.

    <question>
    {question}
    </question>

    Output only the relevant sentences you found, one sentence per line, without any extra characters or explanations.
    </Instructions>
    Assistant:""")

QUESTION_COMPRESS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template="""
    <Instructions>
    <role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

    <task>It is your task to rewrite the following question in a more indirect and compressed form, following these rules:

    <rules>
    1. Make the question more indirect
    2. Make the question shorter
    3. Use abbreviations if possible
    </rules>

    <question>
    {question}
    </question>

    Your output should only be the rewritten question with a question mark "?" at the end. Do not provide any other explanation or text.
    </task>
    </Instructions>
    
    """)

GROUNDEDNESS_CHECK_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context","question"],
    template="""
    <Instructions>
    You will be given a context and a question related to that context.

    Your task is to provide an evaluation of how well the given question can be answered using only the information provided in the context. Rate this on a scale from 1 to 5, where:

    1 = The question cannot be answered at all based on the given context
    2 = The context provides very little relevant information to answer the question
    3 = The context provides some relevant information to partially answer the question 
    4 = The context provides substantial information to answer most aspects of the question
    5 = The context provides all the information needed to fully and unambiguously answer the question

    First, read through the provided context carefully:

    <context>
    {context}
    </context>

    Then read the question:

    <question>
    {question}
    </question>

    Evaluate how well you think the question can be answered using only the context information. Provide your reasoning first in an <evaluation> section, explaining what relevant or missing information from the context led you to your evaluation score in only one sentence.

    Provide your evaluation in the following format:

    <rating>(Your rating from 1 to 5)</rating>
    
    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>


    </Instructions>
    
    """)

RELEVANCE_CHECK_DRP = PromptTemplate(
    input_variables=["question"],
    template="""
    <Instructions>
    You will be given a question related to NASA's Deferred Resignation Program (DRP) and the documentation required for final separation. Your task is to evaluate how useful this question would be for an HR specialist, supervisor, or program administrator involved in managing DRP cases and processing final separation actions.

    To evaluate the usefulness of the question, consider the following criteria:

    1. Relevance: Is the question directly relevant to the Deferred Resignation Program, separation procedures, HR policy, benefits implications, or administrative requirements? Questions unrelated to these areas should receive a lower rating.

    2. Practicality: Does the question address a real HR, personnel, or administrative issue that commonly arises during DRP participation or final separation? Questions with no actionable or procedural value are less useful.

    3. Clarity: Is the question clear, specific, and unambiguous? Vague or poorly defined questions reduce usefulness.

    4. Depth: Does the question require an understanding of NASA HR policy, DRP timelines, eligibility criteria, documentation requirements, or separation workflows? Superficial questions may be less useful.

    5. Applicability: Would answering this question help an HR professional or program administrator improve accuracy, compliance, communication, or efficiency when processing DRP cases? If the real-world applicability is limited, the score should be lower.

    Provide your evaluation in the following format:

    <rating>(Your rating from 1 to 5)</rating>

    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>

    Here is an example:
    <evaluation>The question is very relevant because it asks about specific DRP separation steps that HR must verify during final processing.</evaluation>
    <rating>5</rating>

    Here is the question:

    {question}
    </Instructions>
    """)

RELEVANCE_CHECK_CFR = PromptTemplate(
    input_variables=["question"],
    template="""
    <Instructions>
    You will be given a question related to the Code of Federal Regulations (CFR) Title 5, which covers Government Organization and Employees. Your task is to evaluate how useful this question would be for various stakeholders including HR specialists, managers, federal employees, soon-to-be retirees, or anyone researching federal employment policies and regulations.
    
    To evaluate the usefulness of the question, consider the following criteria:
    1. Relevance: Is the question directly relevant to federal employment policies, procedures, benefits, rights, obligations, retirement, leave, pay, performance management, or other Title 5 topics? Questions unrelated to these areas should receive a lower rating.
    2. Practicality: Does the question address a real concern that commonly arises in federal employment contexts? Questions with no actionable value or practical application are less useful.
    3. Communication Style: The question may be phrased in various ways - from formal/technical language to casual/conversational style. Both should be considered equally valid if the underlying information need is clear. Don't penalize conversational phrasing if the intent can be understood.
    4. Information Need: Does the question express a genuine need for information related to Title 5 regulations? Consider whether the question seeks substantive knowledge about policies, procedures, interpretations, or applications, regardless of how it's phrased. Both basic and complex questions are valuable if they represent real information needs.
    5. Applicability: Would answering this question help federal employees, HR professionals, managers, or retirees navigate policies, make informed decisions, ensure compliance, or understand their rights and responsibilities? If the real-world applicability is limited, the score should be lower.
    
    Provide your evaluation in the following format:
    <rating>(Your rating from 1 to 5)</rating>
    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>
    
    Here is an example:
    <evaluation>The question is highly relevant because it addresses a specific retirement eligibility requirement that affects planning decisions for federal employees and requires HR specialists to accurately interpret when processing retirement applications.</evaluation>
    <rating>5</rating>
    
    Here is the question:
    {question}
    </Instructions>
    """)

RELEVANCE_CHECK_HR = PromptTemplate(
    input_variables=["question"],
    template="""
    <Instructions>
    You will be given a question related to the concepts in *The HR Scorecard: Linking People, Strategy, and Performance*. Your task is to evaluate how useful this question would be for an HR specialist, leader, or organizational strategist applying the principles from the book.

    To evaluate usefulness, consider the following criteria:
    1. Relevance: Is the question directly relevant to HR strategy, HR measurement, HR architecture, High-Performance Work Systems (HPWS), strategic alignment, value creation, or the HR Scorecard model? Questions unrelated to these areas should receive a lower rating.
    2. Practicality: Does the question address real organizational, HR, or measurement challenges that commonly arise when implementing an HR Scorecard or strategic HR system?
    3. Clarity: Is the question clear, specific, and unambiguous? Vague or generic questions reduce usefulness.
    4. Depth: Does the question require understanding HR measurement systems, causal models, leading vs. lagging indicators, alignment, competencies, or strategy implementation? Superficial questions may be less useful.
    5. Applicability: Would answering this question help an HR professional, strategist, or leader improve measurement accuracy, HR-system alignment, or strategic impact using HR Scorecard principles?

    Provide your evaluation in the following format:
    <rating>(Your rating from 1 to 5)</rating>
    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>

    Example:
    <evaluation>The question is highly relevant because it asks directly about aligning HR practices to strategy, which is central to the HR Scorecard framework.</evaluation>
    <rating>5</rating>

    Here is the question:
    {question}
    </Instructions>
    """)

RELEVANCE_CHECK_SWA = PromptTemplate(
    input_variables=["question"],
    template="""
    <Instructions>
    You will be given a question related to Strategic Workforce Analytics. Your task is to evaluate how useful this question would be for various stakeholders including HR professionals, people analytics specialists, business leaders, workforce planners, or anyone researching how to effectively implement and use workforce analytics.
    
    To evaluate the usefulness of the question, consider the following criteria:
    1. Relevance: Is the question directly relevant to workforce analytics methodology, strategic implementation, data analysis techniques, organizational capabilities, or practical applications? Questions unrelated to these areas should receive a lower rating.
    2. Practicality: Does the question address a real challenge or interest that commonly arises when implementing or using workforce analytics? Questions with no actionable value or practical application are less useful.
    3. Communication Style: The question may be phrased in various ways - from technical/academic language to casual/conversational style. Both should be considered equally valid if the underlying information need is clear. Don't penalize conversational phrasing if the intent can be understood.
    4. Information Need: Does the question express a genuine need for information that would be found in a strategic workforce analytics document? Consider whether the question seeks insights about methodologies, building capabilities, applications, or best practices, regardless of how it's phrased. Both basic and complex questions are valuable if they represent real information needs.
    5. Applicability: Would answering this question help analytics professionals, HR leaders, or business executives better understand, implement, or leverage workforce analytics for strategic impact? If the real-world applicability is limited, the score should be lower.
    
    Provide your evaluation in the following format:
    <rating>(Your rating from 1 to 5)</rating>
    <evaluation>(Your evaluation and reasoning for the rating)</evaluation>
    
    Here is an example:
    <evaluation>This question about integrating workforce analytics with business strategy addresses a core challenge faced by organizations implementing analytics programs. It seeks practical guidance on alignment that would benefit analytics teams and business leaders alike, making it highly relevant to the document's content on strategic impact.</evaluation>
    <rating>5</rating>
    
    Here is the question:
    {question}
    </Instructions>
    """)