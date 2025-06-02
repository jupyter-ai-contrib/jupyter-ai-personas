# The Finance Persona

This persona contains examples that showcase financial uses of agentic workflows. It uses APIs and tools from [financial datasets](https://www.financialdatasets.ai). It is based on a multi-agent system that uses the [Agno library](https://github.com/agno-agi/agno) under the hood. The persona is adept at responding to focused natural language instructions to perform various financial analyses and tasks such as collecting data, pricing securities, analyzing SEC filings, forecasting, etc. The Persona is extendable with many additional financial analyses.  

The Persona is packaged in a folder titled `financePersona` as containing the following files:

1. `finance_persona.py`: contains the main code for the persona. 
2. `fd.py`: This contains the `FinancialDatasetsTools` class with several functions and tools.
3. `finance_requirements.txt`: for installing the various dependencies. 
4. This `finance_README.md` file for documenting the finance persona. 
5. Various image files for the documentation.

## Set up

The use of this persona requires an API key from Financial Datasets. You can register and find your account to get an API key. It has excellent tools for access to various quantitative and textual data. For details, see the [documentation](https://docs.financialdatasets.ai/introduction). Once you have the API key, enter AI Settings and choose any `TogetherAI` model, when asked for the API key, enter the Financial Datasets API key, as this persona uses that as a placeholder for the datasets API key. 

Update the TOML file `jupyter_ai_test/pyproject.toml` in Jupyter AI with the following line in the personas section:

```
finance-persona = "jupyter_ai_test.financePersona.finance_persona:FinancePersona"
```

Install the dependencies using the requirements file metioned above as follows:
```
jupyter-ai> pip install -r packages/jupyter-ai-test/jupyter_ai_test/financePersona/finance_requirements.txt
```

Make sure you have API keys for the models used by Agno, see the code in `persona_finance.py` where the code used is based on OpenAI's `gpt-4.1` model:
```py
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
```

For reference to usage of LLMs woth Agno, see https://docs.agno.com/models/introduction. 


## Examples

When the Persona is called upon it examines the query and determines whether it is related to finance or not. If it is financial, then it hands it off to the multi-agent system to process. If not, it will attempt to generate an answer using the chosen LLM's memory. This may be modified of course to a refusal to respond. 

### Example 1

Ask the agentic system to describe what tools it uses. 

![Financial Tools](financial_tools.png)

### Example 2

Ask the agentic system to summarize the latest SEC 10-K filing from any company. 

![SEC Tools](sec_10k.png)

### Example 3

The agent can download stock prices and analyze them. 

![Stock price Tools](avg_stk_pr.png)

### Example 4

The agentic system can be used for forecasting using ARIMA models. 

![Forecasting Tools](arima.png)