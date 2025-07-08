from utils.llm import OpenAILLM, DeepSeekLLM
from utils.prompts import SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re

class Summarizer:
    def __init__(self, llm_summarize):
        self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        self.llm_summarize = llm_summarize

        self.llm = OpenAILLM()
        if(self.llm_summarize == "DeepSeekLLM"):
            self.llm = DeepSeekLLM()

        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

    def get_summary(self, ticker, tweets):
        summary = None
        if tweets != []:

            # print("tweets len = ", len(tweets))
            if(len(tweets) > 200):
                tweets = tweets[:200]

            print("tweets len = ", len(tweets))
            

            prompt = self.summarize_prompt.format(
                                    ticker = ticker,
                                    examples = self.summarize_examples,
                                    tweets = "\n".join(tweets))

            summary = ""
            if (self.llm_summarize == "DeepSeekLLM"):
                print("self.llm_summarize == DeepSeekLLM")
                summary = self.llm(prompt)
            else: # self.llm_summarize == "OpenAILLM"
                # print("self.llm_summarize == OpenAILLM")
                # print(len(self.enc.encode(prompt)))
                # while len(self.enc.encode(prompt)) > 16385:
                #     tweets = tweets[:-1]
                prompt = self.summarize_prompt.format(
                                        ticker = ticker,
                                        examples = self.summarize_examples,
                                        tweets = "\n".join(tweets))

                summary = self.llm(prompt)

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)
