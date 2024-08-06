import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
import time
import re

from openai import OpenAI
from together import Together
import anthropic


API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

MODELS = dict(
    {
        "openai/gpt-3.5-turbo": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-3.5-turbo",
        },
        "openai/gpt-4o-mini": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-4o-mini",
        },
        "openai/gpt-4o": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-4o",
        },
        "meta/llama-3.1-8b": {
            "company": "meta",
            "model_class": "TogetherModel",
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        },
        "meta/llama-3.1-70b": {
            "company": "meta",
            "model_class": "TogetherModel",
            "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        },
        "meta/llama-3.1-405b": {
            "company": "meta",
            "model_class": "TogetherModel",
            "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        },
        "mistral/mixtral-8x7b": {
            "company": "mistral",
            "model_class": "TogetherModel",
            "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        },
        "anthropic/claude-3-haiku": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-3-haiku-20240307",
        },
        "anthropic/claude-3.5-sonnet": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-3-5-sonnet-20240620",
        },
    }
)

def get_api_key(company_identifier: str) -> str:
    """
    Helper Function to retrieve API key from files
    """
    path_key = f"../api_keys/{company_identifier}_key.txt"

    if os.path.exists(path_key):
        with open(path_key, encoding="utf-8") as f:
            key = f.read()
        return key

    raise ValueError(f"API KEY not available at: {path_key}")

def get_timestamp():
    """
    Generate timestamp of format Y-M-D_H:M:S
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


####################################################################################
# MODEL WRAPPERS
####################################################################################


class LanguageModel:
    """ Generic LanguageModel Class"""

    def __init__(self, model_name):
        assert model_name in MODELS, f"Model {model_name} is not supported!"

        # Set some default model variables
        self._model_id = model_name
        self._model_name = MODELS[model_name]["model_name"]
        self._company = MODELS[model_name]["company"]

    def get_model_id(self):
        """Return model_id"""
        return self._model_id
    
    def get_model_company(self):
        return self._company

    def get_greedy_answer(
        self, messages: List[Dict], max_tokens: int
    ) -> str:
        """
        Gets greedy answer for prompt_base

        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :return:                answer string
        """

    def get_top_p_answer(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Gets answer using sampling (based on top_p and temperature)

        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :param max_tokens       max tokens in answer
        :param temperature      temperature for top_p sampling
        :param top_p            top_p parameter
        :return:                answer string
        """


class OpenAIModel(LanguageModel):
    """OpenAI API Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OpenAIModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("openai")
        self.openai = OpenAI(api_key = api_key)

    def _prompt_request(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: int = 0,
        stop: List = ["Human:", " AI:"],
        echo: bool = False,
    ):
        success = False
        t = 0

        MAX_RETRIES = 5
        response = None

        while not success and t < MAX_RETRIES:
            try:
                # Query ChatCompletion endpoint
                response = self.openai.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )

                # Set success flag
                success = True

            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS) - 1)

        return response

    def get_greedy_answer(
        self, messages:List[Dict], max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # (1) Top-P Sampling
        response = self._prompt_request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=1,
            stop=["Human:", " AI:"],
            echo=False,
        )

        completion = response.choices[0].message.content.strip()

        result["answer"] = completion.strip()

        return result

class TogetherModel(LanguageModel):
    """Together API Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "TogetherModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("together")
        self.together = Together(api_key = api_key)

    def _prompt_request(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: int = 0, # changed
        stop: List = ["Human:", " AI:"],
        echo: bool = False,
    ):
        success = False
        t = 0

        MAX_RETRIES = 5
        response = None

        while not success and t < MAX_RETRIES:
            try:
                # Query ChatCompletion endpoint
                response = self.together.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )

                # Set success flag
                success = True

            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS) - 1)

        return response

    def get_greedy_answer(
        self, messages:List[Dict], max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # (1) Top-P Sampling
        response = self._prompt_request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=1,
            stop=["Human:", " AI:"],
            echo=False,
        )

        completion = response.choices[0].message.content.strip()

        result["answer"] = completion.strip()

        return result
    
class AnthropicModel(LanguageModel):
    """Anthropic API Wrapper"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "AnthropicModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        api_key = get_api_key("anthropic")
        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

    def _prompt_request(
        self,
        messages: List[Dict],
        system: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        success = False
        t = 0

        while not success and t < len(API_TIMEOUTS):
            try:
                response = self._anthropic_client.messages.create(
                    model=self._model_name,
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                success = True
            except Exception as e:
                print(f"API call failed: {e}")
                time.sleep(API_TIMEOUTS[t])
                t += 1

        if not success:
            raise Exception("Failed to get response from Anthropic API after multiple retries")

        return response

    def get_top_p_answer(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        system: str = "",
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        response = self._prompt_request(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        result["answer"] = response.content[0].text.strip()
        return result

    def get_greedy_answer(
        self, messages: List[Dict], max_tokens: int, system: str = ""
    ) -> str:
        return self.get_top_p_answer(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )
    
####################################################################################
# MODEL CREATOR
####################################################################################


def create_model(model_name):
    """Init Models from model_name only"""
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model_name)

    raise ValueError(f"Unknown Model '{model_name}'")