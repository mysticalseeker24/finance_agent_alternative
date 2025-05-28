"""Analysis Agent for performing quantitative financial analysis."""

from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

from agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent for performing quantitative financial analysis."""

    def __init__(self):
        """Initialize the Analysis agent."""
        super().__init__("Analysis Agent")
        logger.info("Analysis Agent initialized")

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an analysis request.

        Args:
            request: The request containing operation and parameters.
                    Expected format:
                    {
                        "operation": "calculate_risk"|"analyze_portfolio"|"analyze_performance"|"detect_anomalies",
                        "parameters": {...}  # Operation-specific parameters
                    }

        Returns:
            The analysis result.
        """
        operation = request.get("operation")
        parameters = request.get("parameters", {})

        if not operation:
            return {"error": "No operation specified"}

        # Execute the requested operation
        if operation == "calculate_risk":
            data = parameters.get("data")
            tickers = parameters.get("tickers")
            period = parameters.get("period")

            if not data and not tickers:
                return {
                    "error": "Either data or tickers must be provided for risk calculation"
                }

            result = await self._calculate_risk(data, tickers, period)
            return {"data": result}

        elif operation == "analyze_portfolio":
            portfolio = parameters.get("portfolio")
            market_data = parameters.get("market_data")

            if not portfolio:
                return {"error": "No portfolio provided for analysis"}

            result = await self._analyze_portfolio(portfolio, market_data)
            return {"data": result}

        elif operation == "analyze_performance":
            data = parameters.get("data")
            benchmark = parameters.get("benchmark")
            period = parameters.get("period")

            if not data:
                return {"error": "No data provided for performance analysis"}

            result = await self._analyze_performance(data, benchmark, period)
            return {"data": result}

        elif operation == "detect_anomalies":
            data = parameters.get("data")
            threshold = parameters.get("threshold", 2.0)

            if not data:
                return {"error": "No data provided for anomaly detection"}

            result = await self._detect_anomalies(data, threshold)
            return {"data": result}

        else:
            return {"error": f"Unknown operation: {operation}"}

    async def _calculate_risk(
        self,
        data: Optional[Dict[str, Any]] = None,
        tickers: Optional[List[str]] = None,
        period: Optional[str] = "1y",
    ) -> Dict[str, Any]:
        """Calculate risk metrics for stocks or portfolios.

        Args:
            data: Historical price data for stocks.
            tickers: List of stock tickers to analyze.
            period: Time period for analysis.

        Returns:
            Risk metrics.
        """
        try:
            # If data is provided, use it directly
            if data:
                # Convert data to DataFrame if it's not already
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
            else:
                # This is a placeholder - in a real implementation, you would fetch data for tickers
                logger.warning("No data provided, using placeholder data")
                dates = pd.date_range(end=datetime.now(), periods=252, freq="B")
                data = pd.DataFrame(
                    {"Date": dates, "Close": np.random.normal(100, 2, 252)}
                )

            # Calculate daily returns
            if "Close" in data.columns and len(data) > 1:
                data["Return"] = data["Close"].pct_change().dropna()

            # Calculate risk metrics
            volatility = float(
                data["Return"].std() * np.sqrt(252)
            )  # Annualized volatility
            sharpe_ratio = float(
                data["Return"].mean() / data["Return"].std() * np.sqrt(252)
            )  # Annualized Sharpe Ratio
            max_drawdown = float(
                (1 - data["Close"] / data["Close"].cummax()).max()
            )  # Maximum drawdown
            var_95 = float(
                np.percentile(data["Return"].dropna(), 5)
            )  # 95% Value at Risk

            # Calculate beta if benchmark data is available (placeholder for now)
            beta = 1.0  # Placeholder

            result = {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "value_at_risk_95": var_95,
                "beta": beta,
            }

            logger.info(f"Calculated risk metrics: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calculating risk: {str(e)}")
            return {"error": str(e)}

    async def _analyze_portfolio(
        self,
        portfolio: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a portfolio's composition and risk.

        Args:
            portfolio: List of portfolio items with ticker and weight.
                      Example: [{"ticker": "AAPL", "weight": 0.25}, ...]
            market_data: Market data for the portfolio stocks.

        Returns:
            Portfolio analysis.
        """
        try:
            # Check portfolio structure
            if not portfolio or not isinstance(portfolio, list):
                return {"error": "Invalid portfolio structure"}

            # Extract tickers and weights
            tickers = [item.get("ticker") for item in portfolio if item.get("ticker")]
            weights = [
                item.get("weight", 0) for item in portfolio if item.get("ticker")
            ]

            # Normalize weights if they don't sum to 1
            weight_sum = sum(weights)
            if weight_sum != 1.0:
                weights = [w / weight_sum for w in weights]

            # If market data is provided, use it for analysis
            # Otherwise, we can only analyze the portfolio composition
            sector_exposure = {}
            country_exposure = {}
            asset_type_exposure = {}

            # In a real implementation, you would use market_data to determine these
            # For now, we'll create placeholder data
            sector_map = {
                "AAPL": "Technology",
                "MSFT": "Technology",
                "AMZN": "Consumer Cyclical",
                "GOOGL": "Communication Services",
                "BRK.B": "Financial Services",
                "JNJ": "Healthcare",
                "JPM": "Financial Services",
                "V": "Financial Services",
                "PG": "Consumer Defensive",
                "UNH": "Healthcare",
            }

            country_map = {
                "AAPL": "United States",
                "MSFT": "United States",
                "AMZN": "United States",
                "GOOGL": "United States",
                "BRK.B": "United States",
                "JNJ": "United States",
                "JPM": "United States",
                "V": "United States",
                "PG": "United States",
                "UNH": "United States",
            }

            asset_type_map = {
                "AAPL": "Equity",
                "MSFT": "Equity",
                "AMZN": "Equity",
                "GOOGL": "Equity",
                "BRK.B": "Equity",
                "JNJ": "Equity",
                "JPM": "Equity",
                "V": "Equity",
                "PG": "Equity",
                "UNH": "Equity",
            }

            # Calculate exposures
            for ticker, weight in zip(tickers, weights):
                sector = sector_map.get(ticker, "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

                country = country_map.get(ticker, "Unknown")
                country_exposure[country] = country_exposure.get(country, 0) + weight

                asset_type = asset_type_map.get(ticker, "Unknown")
                asset_type_exposure[asset_type] = (
                    asset_type_exposure.get(asset_type, 0) + weight
                )

            # Format exposures as lists for easier processing
            sector_list = [
                {"sector": sector, "weight": weight}
                for sector, weight in sector_exposure.items()
            ]

            country_list = [
                {"country": country, "weight": weight}
                for country, weight in country_exposure.items()
            ]

            asset_type_list = [
                {"asset_type": asset_type, "weight": weight}
                for asset_type, weight in asset_type_exposure.items()
            ]

            # Calculate concentration metrics
            concentration = sum([w * w for w in weights])  # Herfindahl-Hirschman Index

            # Determine if portfolio is diversified based on concentration
            diversification_status = (
                "Well Diversified"
                if concentration < 0.15
                else (
                    "Moderately Diversified" if concentration < 0.25 else "Concentrated"
                )
            )

            result = {
                "tickers": tickers,
                "weights": weights,
                "sector_exposure": sector_list,
                "country_exposure": country_list,
                "asset_type_exposure": asset_type_list,
                "concentration": concentration,
                "diversification_status": diversification_status,
            }

            # If market data is available, calculate portfolio risk metrics
            if market_data:
                # Placeholder for risk calculations
                result["risk_metrics"] = {
                    "portfolio_volatility": 0.15,  # Placeholder
                    "portfolio_sharpe": 1.2,  # Placeholder
                    "portfolio_var_95": -0.02,  # Placeholder
                    "portfolio_max_drawdown": 0.25,  # Placeholder
                }

            logger.info(f"Analyzed portfolio with {len(tickers)} stocks")
            return result

        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return {"error": str(e)}

    async def _analyze_performance(
        self,
        data: Dict[str, Any],
        benchmark: Optional[Dict[str, Any]] = None,
        period: Optional[str] = "1y",
    ) -> Dict[str, Any]:
        """Analyze performance of a stock or portfolio.

        Args:
            data: Historical price data.
            benchmark: Benchmark data for comparison.
            period: Time period for analysis.

        Returns:
            Performance analysis.
        """
        try:
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Ensure 'Close' column exists
            if "Close" not in data.columns:
                return {"error": "Data must contain a 'Close' column"}

            # Calculate returns
            data["Return"] = data["Close"].pct_change().dropna()

            # Calculate performance metrics
            total_return = float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)
            annualized_return = (
                float((1 + total_return) ** (252 / len(data)) - 1)
                if len(data) > 1
                else 0
            )
            volatility = float(data["Return"].std() * np.sqrt(252))
            sharpe_ratio = (
                float(data["Return"].mean() / data["Return"].std() * np.sqrt(252))
                if volatility > 0
                else 0
            )
            max_drawdown = float((1 - data["Close"] / data["Close"].cummax()).max())

            # Calculate monthly returns
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"])
                data["Year"] = data["Date"].dt.year
                data["Month"] = data["Date"].dt.month
                monthly_returns = (
                    data.groupby(["Year", "Month"])["Return"].sum().reset_index()
                )
                monthly_returns_list = monthly_returns["Return"].tolist()
            else:
                monthly_returns_list = []

            result = {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "monthly_returns": monthly_returns_list,
            }

            # Compare with benchmark if provided
            if benchmark:
                # Convert benchmark to DataFrame if it's not already
                if not isinstance(benchmark, pd.DataFrame):
                    benchmark = pd.DataFrame(benchmark)

                # Ensure 'Close' column exists in benchmark
                if "Close" in benchmark.columns:
                    benchmark["Return"] = benchmark["Close"].pct_change().dropna()
                    benchmark_total_return = float(
                        benchmark["Close"].iloc[-1] / benchmark["Close"].iloc[0] - 1
                    )
                    benchmark_annualized_return = (
                        float(
                            (1 + benchmark_total_return) ** (252 / len(benchmark)) - 1
                        )
                        if len(benchmark) > 1
                        else 0
                    )

                    # Calculate alpha and beta
                    if len(data) == len(benchmark) and len(data) > 1:
                        # Ensure both datasets have the same length for regression
                        x = benchmark["Return"].values
                        y = data["Return"].values
                        beta = (
                            float(np.cov(y, x)[0, 1] / np.var(x))
                            if np.var(x) > 0
                            else 1.0
                        )
                        alpha = float(np.mean(y) - beta * np.mean(x))
                        alpha_annualized = float((1 + alpha) ** 252 - 1)

                        result["relative_performance"] = {
                            "alpha": alpha_annualized,
                            "beta": beta,
                            "excess_return": total_return - benchmark_total_return,
                            "benchmark_return": benchmark_total_return,
                            "benchmark_annualized_return": benchmark_annualized_return,
                        }

            logger.info(f"Analyzed performance over {len(data)} data points")
            return result

        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {"error": str(e)}

    async def _detect_anomalies(
        self, data: Dict[str, Any], threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in financial data.

        Args:
            data: Financial data to analyze.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            List of detected anomalies.
        """
        try:
            # Convert data to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # Ensure required columns exist
            if "Close" not in data.columns:
                return [{"error": "Data must contain a 'Close' column"}]

            # Calculate returns
            data["Return"] = data["Close"].pct_change().dropna()

            # Calculate z-scores for returns
            mean_return = data["Return"].mean()
            std_return = data["Return"].std()
            data["Z_Score"] = (
                (data["Return"] - mean_return) / std_return if std_return > 0 else 0
            )

            # Detect anomalies based on z-score threshold
            anomalies = data[abs(data["Z_Score"]) > threshold].copy()

            # Format anomalies for return
            anomalies_list = []
            for _, row in anomalies.iterrows():
                anomaly = {
                    "date": row["Date"].isoformat() if "Date" in row else None,
                    "close": float(row["Close"]),
                    "return": float(row["Return"]),
                    "z_score": float(row["Z_Score"]),
                    "direction": "up" if row["Return"] > 0 else "down",
                    "magnitude": (
                        "extreme" if abs(row["Z_Score"]) > 3.0 else "significant"
                    ),
                }
                anomalies_list.append(anomaly)

            logger.info(
                f"Detected {len(anomalies_list)} anomalies with threshold {threshold}"
            )
            return anomalies_list

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return [{"error": str(e)}]

    async def calculate_risk(
        self,
        data: Optional[Dict[str, Any]] = None,
        tickers: Optional[List[str]] = None,
        period: Optional[str] = "1y",
    ) -> Dict[str, Any]:
        """Calculate risk metrics for stocks or portfolios.

        Args:
            data: Historical price data for stocks.
            tickers: List of stock tickers to analyze.
            period: Time period for analysis.

        Returns:
            Risk metrics.
        """
        request = {
            "operation": "calculate_risk",
            "parameters": {"data": data, "tickers": tickers, "period": period},
        }
        return await self.run(request)

    async def analyze_portfolio(
        self,
        portfolio: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a portfolio's composition and risk.

        Args:
            portfolio: List of portfolio items with ticker and weight.
            market_data: Market data for the portfolio stocks.

        Returns:
            Portfolio analysis.
        """
        request = {
            "operation": "analyze_portfolio",
            "parameters": {"portfolio": portfolio, "market_data": market_data},
        }
        return await self.run(request)

    async def analyze_performance(
        self,
        data: Dict[str, Any],
        benchmark: Optional[Dict[str, Any]] = None,
        period: Optional[str] = "1y",
    ) -> Dict[str, Any]:
        """Analyze performance of a stock or portfolio.

        Args:
            data: Historical price data.
            benchmark: Benchmark data for comparison.
            period: Time period for analysis.

        Returns:
            Performance analysis.
        """
        request = {
            "operation": "analyze_performance",
            "parameters": {"data": data, "benchmark": benchmark, "period": period},
        }
        return await self.run(request)

    async def detect_anomalies(
        self, data: Dict[str, Any], threshold: float = 2.0
    ) -> Dict[str, Any]:
        """Detect anomalies in financial data.

        Args:
            data: Financial data to analyze.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            Detected anomalies.
        """
        request = {
            "operation": "detect_anomalies",
            "parameters": {"data": data, "threshold": threshold},
        }
        return await self.run(request)
