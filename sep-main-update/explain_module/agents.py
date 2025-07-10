from typing import List, Union, Literal
from utils.llm import OpenAILLM, NShotLLM, DeepSeekLLM #, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES

# LLM_EXPLAIN = OpenAILLM()

class PredictAgent:
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 technical_indicators: str,
                 predict_llm = OpenAILLM()
                 ) -> None:

        self.ticker = ticker
        self.summary = summary
        self.technical_indicators = technical_indicators  # Lưu chỉ báo kỹ thuật
        self.target = target
        self.prediction = ''

        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.llm = predict_llm

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Facts:\n" + self.summary + "\n\nTechnical Indicators:\n" + self.technical_indicators + "\n\nPrice Movement: "
        self.scratchpad += facts
        # print(facts, end="")

        self.scratchpad += self.prompt_agent()
        response = self.scratchpad.split('Price Movement: ')[-1]
        if (self.prediction.lower() not in ['positive', 'negative']):
            for word in response.split('Explanation')[0].split():
                word_lower = word.lower()
                if 'positive' in word_lower:
                    self.prediction = 'Positive'
                    break
                elif 'negative' in word_lower:
                    self.prediction = 'Negative'
                    break

        self.prediction = response.split()[0]
        print(response, end="\n\n\n\n")

        self.finished = True

    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str:
        return self.predict_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            summary = self.summary,
                            technical_indicators=self.technical_indicators)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        print("self.target, self.prediction", self.target," /////////", self.prediction)
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


class PredictReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 technical_indicators: str = "",
                 predict_llm = OpenAILLM(),
                 reflect_llm = OpenAILLM()
                 ) -> None:

        super().__init__(ticker, summary, target, technical_indicators, predict_llm)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        self.reflect_prompt = REFLECT_INSTRUCTION
        self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
        self.reflections = []
        self.reflections_str: str = ''

    def run(self, reset=True) -> None:
        # print("self.is_finished() = ", self.is_finished(), "     ", "not self.is_correct() = ", not self.is_correct())
        if self.is_finished() and not self.is_correct():
            self.reflect()
        PredictAgent.run(self, reset=reset)

    def reflect(self) -> None:
        print('Reflecting...\n')
        reflection = self.prompt_reflection()
        self.reflections += [reflection]
        self.reflections_str = format_reflections(self.reflections)
        print(self.reflections_str, end="\n\n\n\n")

    def prompt_reflection(self) -> str:
        return self.reflect_llm(self._build_reflection_prompt())

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            ticker = self.ticker,
                            scratchpad = self.scratchpad)

    def _build_agent_prompt(self) -> str:
        prompt = self.agent_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            reflections = self.reflections_str,
                            summary = self.summary,
                            technical_indicators=self.technical_indicators)
        return prompt

    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        PredictAgent.run(self, reset=reset)


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def EM(prediction, sentiment) -> bool:
    return prediction.lower() == sentiment.lower()
