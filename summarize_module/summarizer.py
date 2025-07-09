from utils.llm import OpenAILLM, DeepSeekLLM
from utils.prompts import SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re
from openai import BadRequestError  # Thêm để bắt lỗi context

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

            # # print("tweets len = ", len(tweets))
            if(len(tweets) > 175):
                tweets = tweets[:175]

            print("tweets len = ", len(tweets))
            

            prompt = self.summarize_prompt.format(
                                    ticker = ticker,
                                    examples = self.summarize_examples,
                                    tweets = "\n".join(tweets))

            # summary = ""
            if (self.llm_summarize == "DeepSeekLLM"):
                print("self.llm_summarize == DeepSeekLLM")
                summary = self.llm(prompt)
            else: # self.llm_summarize == "OpenAILLM"
                # print("self.llm_summarize == OpenAILLM")
                # print(len(self.enc.encode(prompt)))
                # while len(self.enc.encode(prompt)) > 16385:
                #     tweets = tweets[:-1]
                # prompt = self.summarize_prompt.format(
                #                         ticker = ticker,
                #                         examples = self.summarize_examples,
                #                         tweets = "\n".join(tweets))

                summary = self.llm(prompt)
            # while tweets:
            #     try:
            #         prompt = self.summarize_prompt.format(
            #                         ticker = ticker,
            #                         examples = self.summarize_examples,
            #                         tweets = "\n".join(tweets))
            #         summary = self.llm(prompt)
            #         return summary  # Nếu thành công thì return luôn
            #     except BadRequestError as e:
            #         # Nếu lỗi là vượt token, cắt bớt 25 dòng
            #         if "maximum context length" in str(e):
            #             print(f"Prompt quá dài. Cắt bớt 25 dòng. Còn lại: {len(tweets) - 25}")
            #             tweets = tweets[:-25]
            #             continue
            #         else:
            #             raise e  # Nếu lỗi không phải context length thì raise lại
            

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)
