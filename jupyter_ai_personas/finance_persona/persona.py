from typing import Any
from pydantic import Field, BaseModel

from jupyterlab_chat.models import Message
from jupyter_core.paths import jupyter_data_dir

from jupyter_ai.personas import BasePersona, PersonaDefaults
from jupyter_ai.config_manager import DEFAULT_CONFIG_PATH
from litellm import acompletion

from agno.agent import Agent
from agno.models.aws import AwsBedrock
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.csv_toolkit import CsvTools

import os
import json
import pandas as pd  # requires: pip install pandas
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

from .fd import FinancialDatasetsTools

class UserQueryClassifier(BaseModel): 
    is_finance_related: bool = Field(
        description="Returns True if the request is finance related, False otherwise."
    )


def env_api_keys_from_config(API_KEY_NAME, file_path=DEFAULT_CONFIG_PATH):
    """
    Reads the config.json file at 'file_path' and returns the 'api_keys' dictionary.
    The DEFAULT_CONFIG_PATH is at `~/Library/Jupyter/jupyter_ai/config.json` in Jupyter AI.
    Use AI Settings to set the API keys by adding the API key for the 
    Financial Datasets API by choosing any Together AI model and populating the 
    TOGETHER_API_KEY. 
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    for key, value in data.get('api_keys', {}).items():
        os.environ[key] = value
        if key == API_KEY_NAME:
            fin_key = value
    return fin_key


class FinancePersona(BasePersona):
    """
    The Finance persona, using natural language to request financial functionality.
    Uses : Financial Datasets, https://www.financialdatasets.ai/
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @property
    def defaults(self):
        return PersonaDefaults(
            name="FinancePersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="The finance agent provided by Jupyter AI. Tools are in `fd.py`.",
            system_prompt="...",
        )


    async def process_message(self, message: Message):
        if not self.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return

        model_id = self.config_manager.chat_model
        model_args = self.config_manager.chat_model_args
        
        # Check if the prompt is about finance using litellm
        prompt = message.body.split(" ", 1)[1]
        response = await acompletion(
            **model_args,
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a classifier that determines if a message is finance related. Respond with true or false."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=False,
        )
        is_finance = "true" in response.choices[0].message.content.lower()

        # If the message is finance-related, proceed with agentic handling
        if is_finance:
            msg = message.body.split(" ", 1)[1].strip()
            if msg:
                # Call the agno_finance function to process the message
                self.agno_finance(msg)
            else:
                self.send_message("Error: Query failed. Please try again with a different query.")
        else: # If the message is not finance-related, use litellm
            response = await acompletion(
                **model_args,
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": message.body,
                    },
                ],
                stream=False,
            )
            self.send_message(response.choices[0].message.content)

    
    # Use Agno to process financial prompts 
    # Multi agent workflow to get stock prices and forecast them using ARIMA
    def agno_finance(self, message: Message):
        self.send_message("The AGNO Finance agent is processing your request ...")
        FINANCIAL_DATASETS_API_KEY = env_api_keys_from_config(API_KEY_NAME="TOGETHER_API_KEY", file_path=DEFAULT_CONFIG_PATH)
        # Agent for stock prices
        stock_price_agent = Agent(
            role="Get stock prices for a given date range.",
            model=OpenAIChat(id="gpt-4.1"),
            description="Agent to get the stock price information for a ticker.",
            instructions="For a given ticker, please collect the latest stock prices for the date range provided.",
            tools = [
                FinancialDatasetsTools(
                    enable_company_info=False, 
                    enable_prices=True, 
                    api_key=FINANCIAL_DATASETS_API_KEY
                ),
                YFinanceTools(
                    stock_price=True, 
                    analyst_recommendations=True, 
                    stock_fundamentals=True,
                ),
                CsvTools(),
            ],
            show_tool_calls=False,
            markdown=True,
            name = "Stock Price Agent",
        )
        # ARIMA agent to forecast stock prices
        arima_agent = Agent(
            role="Fit an ARIMA model to the stock prices and then forecast the prices for a specified period of time.",
            model=OpenAIChat(id="gpt-4.1"),
            description="Agent to forecast stock prices given time series price information for a ticker.",
            instructions="""
            For a given ticker, please collect the latest closing stock prices for the date range provided by using the `stock_price_agent`.
            Then, fit an ARIMA model to the close stock prices and then forecast the prices for a specified number of periods.
            """,
            tools = [
                FinancialDatasetsTools(
                    enable_company_info=False, 
                    enable_prices=True, 
                    enable_arima=True,
                    api_key=FINANCIAL_DATASETS_API_KEY,
                ),
            ],
            show_tool_calls=False,
            markdown=True, 
            name = "ARIMA Agent",
        )

        # SEC filings agent
        sec_agent = Agent(
            role="Get SEC filings for a given ticker.",
            model=OpenAIChat(id="gpt-4.1"),
            description="Agent to get the SEC filings for a ticker.",
            instructions=[
                "1. For a given ticker, please collect the latest SEC filings, with the URL to the filing.",
                "2. Provide a summary of the filings.",
            ],
            tools = [
                FinancialDatasetsTools(
                    enable_company_info=False, 
                    enable_prices=False, 
                    enable_sec_filings=True, 
                    api_key=FINANCIAL_DATASETS_API_KEY
                ),
                DuckDuckGoTools(),
            ],
            show_tool_calls=False,
            markdown=True,
            name = "SEC Filings Agent",
        )

        # Set up the team of agents
        finance_agent = Team(
            name="Finance Agent Team",
            mode="coordinate", # coordinate or route or collaborate
            members=[stock_price_agent, arima_agent, sec_agent],
            model=OpenAIChat(id="gpt-4.1"),
            description="Team of agents to get stock prices and forecast them using ARIMA.",
            instructions=[
                "You are a team of agents that work together to answer various financial questions.",
                "if asked for stock prices, use the `stock_price_agent`.",
                "If asked to save prices to a CSV file, you will take the prices and then use the `CsvTools` to save them without asking for any further details.",
                "If asked for ARIMA forecast, you will first get the stock prices for the ticker using the `stock_price_agent`."
                "Then, you will use the prices to use the `arima_agent` to create a forecast.",
                "If the request asks for a forecast, you will use both `stock_price_agent` and `arima_agent`.",
                "If the request asks for SEC filings, you will use the `sec_agent`.",
                "If you are not sure about the request, do not ask for clarification. Proceed with the best guess.",
                "Please let the user know that you have finished processing their request.",
            ],
            show_tool_calls=True,
            markdown=True,
        )
        # Run the ic workflow with the message
        response = finance_agent.run(f"{message}")
        if response.content: # in case the response is empty
            response = response.content
        else:
            response = "No response from the Finance Agent. Please try again with a different query."
        self.send_message(response)
