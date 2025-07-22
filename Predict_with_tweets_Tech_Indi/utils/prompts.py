SUMMARIZE_INSTRUCTION = """Given a list of tweets, summarize all key facts regarding {ticker} stock.
Here are some examples:
{examples}
(END OF EXAMPLES)

Tweets:
{tweets}

Facts:"""


PREDICT_INSTRUCTION = """You are given a list of chronological facts spanning multiple days about {ticker} and a set of technical indicators. Your task is to assess the overall impact of these facts and technical indicators on the stock price and provide:

(1) One price Movement: either Positive or Negative no other responses are allowed.  
(2) One explanation: a short, single-paragraph justification summarizing the main reasons behind your prediction.

Do not output anything other than these two parts.
Here is a examples, the contents of the example are not used to deduce the answer:
{examples}
(END OF EXAMPLES)

Facts:
{summary}

Technical Indicators:
{technical_indicators}

Price Movement:"""


PREDICT_REFLECT_INSTRUCTION = """You are given a list of chronological facts spanning multiple days about {ticker} and a set of technical indicators. Your task is to assess the overall impact of these facts and technical indicators on the stock price and provide:

(1) One price Movement: either Positive or Negative no other responses are allowed.  
(2) One explanation: a short, single-paragraph justification summarizing the main reasons behind your prediction.

Do not output anything other than these two parts.
Here is a examples, the contents of the example are not used to deduce the answer:
{examples}
(END OF EXAMPLES)

{reflections}

Facts:
{summary}

Technical Indicators:
{technical_indicators}

Price Movement:"""


REFLECTION_HEADER = 'You have attempted to tackle the following task before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly tackling the given task.\n'


REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to a list of facts and a set of technical indicators to assess their overall impact on the price movement of {ticker} stock. You were unsuccessful in tackling the task because you gave the wrong price movement. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Previous trial:
{scratchpad}

Reflection:"""
