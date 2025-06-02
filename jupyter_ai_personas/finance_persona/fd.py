import os
from os import getenv
from typing import Any, Dict, Optional

import requests
import json

from agno.tools import Toolkit
from agno.utils.log import log_error

from statsmodels.tsa.arima.model import ARIMA


class FinancialDatasetsTools(Toolkit):
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_company_info: bool = True,
        enable_financial_metrics: bool = True,
        enable_financial_statements: bool = True,
        enable_prices: bool = True,
        enable_news: bool = True,
        enable_sec_filings: bool = False,
        enable_arima: bool = True,
        **kwargs,
    ):
        """
        Initialize the Financial Datasets Tools with feature flags.

        Args:
            api_key: API key for Financial Datasets API (optional, can be set via environment variable)
            enable_company_info: Enable company information related functions
            enable_financial_metrics: Enable current and historical financial metrics
            enable_financial_statements: Enable financial statement related functions (income statements, balance sheets, etc.)
            enable_prices: Enable market data related functions (stock prices, earnings, metrics)
            enable_news: Enable news related functions
            enable_sec_filings: Enable SEC filings related functions
        """
        super().__init__(name="financial_datasets_tools", **kwargs)

        self.api_key: Optional[str] = api_key or getenv("FINANCIAL_DATASETS_API_KEY")
        if not self.api_key:
            log_error(
                "FINANCIAL_DATASETS_API_KEY not set. Please set the FINANCIAL_DATASETS_API_KEY environment variable."
            )

        self.base_url = "https://api.financialdatasets.ai"

        if enable_company_info:
            self.register(self.get_company_info)

        if enable_financial_metrics:
            self.register(self.get_historical_metrics)
            self.register(self.get_current_metrics)

        if enable_financial_statements:
            self.register(self.get_income_statements)
            self.register(self.get_balance_sheets)
            self.register(self.get_cash_flow_statements)
            self.register(self.get_all_financial_statements)

        if enable_prices:
            self.register(self.get_stock_prices)

        if enable_news:
            self.register(self.get_news)

        if enable_sec_filings:
            self.register(self.get_sec_filings)

        if enable_arima:
            self.register(self.arima_forecast)  # ARIMA forecasting function


    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Makes a request to the Financial Datasets API.

        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request

        Returns:
            JSON response from the API
        """
        if not self.api_key:
            log_error("No API key provided. Cannot make request.")
            return "API key not set"

        headers = {"X-API-KEY": self.api_key}
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            log_error(f"Error making request to {url}: {str(e)}")
            return f"Error making request to {url}: {str(e)}"

    ###################################
    # Company
    ###################################

    def get_company_info(self, ticker: str) -> str:
        """
        Get company information for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing company information
        """
        params = {"ticker": ticker}
        return self._make_request("company/facts", params)

    ###################################
    # Financial Metrics
    ###################################

    def get_historical_metrics(self,
                               ticker: str,
                               period: str = "ttm",
                               limit: int = 4,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> str:
        """Get historical financial metrics for a ticker.

        Get historical financial metrics for a ticker, including valuation,
        profitability, efficiency, liquidity, leverage, growth, and per share metrics.

        Args:
            ticker: Stock ticker symbol
            period: The time period for the financial data ('annual', 'quarterly', 'ttm')
            limit: The maximum number of results to return (int, default of 4)
            start_date: The start date of the data in the form YYYY-MM-DD (optional)
            end_date: The end date of the data in the form YYYY-MM-DD (optional)
        """
        params = {"ticker": ticker,
                  "period": period,
                  "limit": limit}
        if start_date:
            params['report_period_gte'] = start_date
        if end_date:
            params['report_period_lte'] = end_date
        return self._make_request("financial-metrics", params)

    def get_current_metrics(self, ticker: str) -> str:
        """Get real-time snapshot financial metrics for a ticker.

        Get real-time snapshot current financial metrics for a ticker, including valuation,
        profitability, efficiency, liquidity, leverage, growth, and per share metrics.

        Args:
            ticker: Stock ticker symbol       
        """
        params = {"ticker": ticker}
        return self._make_request("financial-metrics/snapshot", params)

    ###################################
    # Financial Statements
    ###################################

    def get_income_statements(self,
                               ticker: str,
                               period: str = "ttm",
                               limit: int = 4,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> str:
        """Get income statements for a ticker.

        The income statements API provides income statements for a given stock
        ticker. Income statements are financial statements that provide information
        about a company’s revenues, expenses, and profits over a specific period.

        Use this function if you only want income statements and not balance sheets or 
        cash flow statements.

        Args:
            ticker: Stock ticker symbol
            period: The time period for the financial data ('annual', 'quarterly', 'ttm')
            limit: The maximum number of results to return (int, default of 4)
            start_date: The start date of the data in the form YYYY-MM-DD (optional)
            end_date: The end date of the data in the form YYYY-MM-DD (optional)
        """
        params = {"ticker": ticker,
                  "period": period,
                  "limit": limit}
        if start_date:
            params['report_period_gte'] = start_date
        if end_date:
            params['report_period_lte'] = end_date
        return self._make_request("financials/income-statements", params)

    def get_balance_sheets(self,
                               ticker: str,
                               period: str = "ttm",
                               limit: int = 4,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> str:
        """Get balance sheets for a ticker.

        The balance sheets API provides balance sheet data for a given stock
        ticker. Balance sheets are financial statements that summarize a company’s
        assets, liabilities, and shareholders’ equity at a specific point in time.

        Use this function if you only want balance sheets and not income statements or 
        cash flow statements.

        Args:
            ticker: Stock ticker symbol
            period: The time period for the financial data ('annual', 'quarterly', 'ttm')
            limit: The maximum number of results to return (int, default of 4)
            start_date: The start date of the data in the form YYYY-MM-DD (optional)
            end_date: The end date of the data in the form YYYY-MM-DD (optional)
        """
        params = {"ticker": ticker,
                  "period": period,
                  "limit": limit}
        if start_date:
            params['report_period_gte'] = start_date
        if end_date:
            params['report_period_lte'] = end_date
        return self._make_request("financials/balance-sheets", params)

    def get_cash_flow_statements(self,
                               ticker: str,
                               period: str = "ttm",
                               limit: int = 4,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> str:
        """Get cash flow statements for a ticker.

        The cash flow statemenet API provides a company’s cash inflows and outflows
        over a specific period. Cash flow statements are divided into three sections:
        operating activities, investing activities, and financing activities.

        Use this function if you only want cash flow statements and not income statements or
        balance sheets.

        Args:
            ticker: Stock ticker symbol
            period: The time period for the financial data ('annual', 'quarterly', 'ttm')
            limit: The maximum number of results to return (int, default of 4)
            start_date: The start date of the data in the form YYYY-MM-DD (optional)
            end_date: The end date of the data in the form YYYY-MM-DD (optional)
        """
        params = {"ticker": ticker,
                  "period": period,
                  "limit": limit}
        if start_date:
            params['report_period_gte'] = start_date
        if end_date:
            params['report_period_lte'] = end_date
        return self._make_request("financials/cash-flow-statements", params)

    def get_all_financial_statements(self,
                               ticker: str,
                               period: str = "ttm",
                               limit: int = 4,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> str:
        """Get all financial statements for a ticker.

        This endpoint aggregates all financial statements for a ticker into a
        single API call. So, instead of calling 3 endpoints to get income
        statements, balance sheets, and cash flow statements, you can call
        this endpoint once and get all financial statements in one go.

        The endpoint returns the following financial statements:

        * Income Statements
        * Balance Sheets
        * Cash Flow Statements

        Args:
            ticker: Stock ticker symbol
            period: The time period for the financial data ('annual', 'quarterly', 'ttm')
            limit: The maximum number of results to return (int, default of 4)
            start_date: The start date of the data in the form YYYY-MM-DD (optional)
            end_date: The end date of the data in the form YYYY-MM-DD (optional)
        """
        params = {"ticker": ticker,
                  "period": period,
                  "limit": limit}
        if start_date:
            params['report_period_gte'] = start_date
        if end_date:
            params['report_period_lte'] = end_date
        return self._make_request("financials", params)

    ###################################
    # Stock prices
    ###################################

    def get_stock_prices(self,
                         ticker: str,
                         start_date: str,
                         end_date: str,
                         interval: str = "day",
                         interval_multiplier: int = 1,
                         limit: int = 5000) -> dict:
        """
        Get stock prices for a ticker with a start date and end date.

        Args:
            ticker: Stock ticker symbol, such as AAPL
            start_date: The start date for the price data (format: YYYY-MM-DD)
            end_date: The end date for the price data (format: YYYY-MM-DD)
            interval: The time interval for the price data ('second', 'minute', 'day', 'week', 'month', 'year')
            interval_multiplier: The multiplier for the interval, must be >=1
            limit: The maximum number of price records to return (default: 5000, max: 5000)

        Returns:
            Dictionary containing stock prices
        """
        print("FinTools: get_stock_prices ... PIPELINE LOADED")
        params = {"ticker": ticker,
                  "start_date": start_date,
                  "end_date": end_date,
                  "interval": interval,
                  "interval_multiplier": interval_multiplier,
                  "limit": limit}
        return self._make_request("prices", params)

    ###################################
    # Financial News
    ###################################

    def get_news(self,
                ticker: str,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                limit: int = 100) -> str:
        """
        Get real-time and historical news for a ticker.

        The News API lets you pull recent and historical news articles for a
        given ticker. The data is great for understanding the latest news for
        a given ticker and how the sentiment for a ticker has changed over time.
        Our news articles are sourced directly from publishers like The Motley Fool,
        Investing.com, Reuters, and more. The articles are sourced from RSS feeds.

        Args:
            ticker: Stock ticker symbol (optional)
            start_date: The start date for the news data (format: YYYY-MM-DD).
            end_date: The end date for the news data (format: YYYY-MM-DD).
            limit: The maximum number of news articles to return (default: 100, max: 100).

        Returns:
            Dictionary containing news items
        """
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._make_request("news", params)

    ###################################
    # SEC Filings
    ###################################

    def get_sec_filings(self,
                        ticker: str,
                        filing_type: str = '10-K') -> str:
        """
        Get SEC filings (date, URL) for a ticker.

        The Filings endpoint allows you to fetch a list of filings for a
        given company. The endpoint returns all of the filings that the
        company has filed with the SEC. This includes 10-Ks, 10-Qs, 8-Ks, and more.

        Args:
            ticker: Stock ticker symbol
            form_type: The type of filing ('10-K', '10-Q', '8-K', '4', '144') 
        Returns:
            Dictionary containing SEC filings
        """
        print("FinTools: get_sec_filings ... PIPELINE LOADED")
        params: Dict[str, Any] = {"ticker": ticker}
        if filing_type:
            params["filing_type"] = filing_type
        return self._make_request("filings", params)


    ###################################
    # Non FD, non API Models
    ###################################

    # ARIMA forecasting function
    def arima_forecast(self, series, prediction_length, p=61, d=1, q=1):
        """
        Forecasts future data using the ARIMA model.
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
        Args:
            series: Time series data as a list or numpy array
            prediction_length: Number of steps to forecast
            p: autoregressive parameter 
            d: differencing parameter d 
            q: moving average parameter q
        (p,d,q) is the order of the model for the autoregressive, differences, and 
        moving average components.
        Returns:
            forecast: Forecasted values as a numpy array
        """
        print("FinTools: ARIMA ... PIPELINE LOADED")
        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=prediction_length)
        forecast = json.dumps({"forecast": list(forecast)})  # Convert to JSON string for consistency
        print("FinTools: ARIMA ... FORECAST COMPLETED")
        return forecast