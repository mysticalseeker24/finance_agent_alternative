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
            asset_historical_prices_df = parameters.get("asset_historical_prices")
            benchmark_historical_prices_df = parameters.get("benchmark_historical_prices")
            risk_free_rate_param = parameters.get("risk_free_rate", 0.0)

            if asset_historical_prices_df is None or not isinstance(asset_historical_prices_df, pd.DataFrame):
                return {"error": "Asset historical prices (DataFrame) must be provided for risk calculation"}
            if benchmark_historical_prices_df is not None and not isinstance(benchmark_historical_prices_df, pd.DataFrame):
                return {"error": "Benchmark historical prices must be a DataFrame if provided"}

            result = await self._calculate_risk(
                asset_historical_prices=asset_historical_prices_df,
                benchmark_historical_prices=benchmark_historical_prices_df,
                risk_free_rate=risk_free_rate_param
            )
            return {"data": result}

        elif operation == "analyze_portfolio":
            portfolio_composition = parameters.get("portfolio") # The list of {"ticker": "X", "weight": Y}
            asset_metadata_dict = parameters.get("asset_metadata") # The dict of {"TICKER": stock_info_dict}
            portfolio_asset_historical_prices_df = parameters.get("portfolio_asset_historical_prices") # DataFrame of prices
            benchmark_historical_prices_df = parameters.get("benchmark_historical_prices") # Optional DataFrame
            risk_free_rate_param = parameters.get("risk_free_rate", 0.0)

            if not portfolio_composition:
                return {"error": "Portfolio composition list must be provided for analysis"}
            if portfolio_asset_historical_prices_df is None or not isinstance(portfolio_asset_historical_prices_df, pd.DataFrame):
                return {"error": "Portfolio asset historical prices (DataFrame) must be provided"}
            if benchmark_historical_prices_df is not None and not isinstance(benchmark_historical_prices_df, pd.DataFrame):
                return {"error": "Benchmark historical prices must be a DataFrame if provided"}

            result = await self._analyze_portfolio(
                portfolio=portfolio_composition,
                asset_metadata=asset_metadata_dict,
                portfolio_asset_historical_prices=portfolio_asset_historical_prices_df,
                benchmark_historical_prices=benchmark_historical_prices_df,
                risk_free_rate=risk_free_rate_param
            )
            return {"data": result}

        elif operation == "analyze_performance":
            asset_historical_prices_df = parameters.get("asset_historical_prices")
            benchmark_historical_prices_df = parameters.get("benchmark_historical_prices")
            risk_free_rate_param = parameters.get("risk_free_rate", 0.0)

            if asset_historical_prices_df is None or not isinstance(asset_historical_prices_df, pd.DataFrame):
                return {"error": "Asset historical prices (DataFrame) must be provided for performance analysis"}
            if benchmark_historical_prices_df is not None and not isinstance(benchmark_historical_prices_df, pd.DataFrame):
                return {"error": "Benchmark historical prices must be a DataFrame if provided"}
                
            result = await self._analyze_performance(
                asset_historical_prices=asset_historical_prices_df,
                benchmark_historical_prices=benchmark_historical_prices_df,
                risk_free_rate=risk_free_rate_param
            )
            return {"data": result}

        elif operation == "detect_anomalies":
            asset_historical_prices_df = parameters.get("asset_historical_prices")
            threshold_param = parameters.get("threshold", 2.0)

            if asset_historical_prices_df is None or not isinstance(asset_historical_prices_df, pd.DataFrame):
                return {"error": "Asset historical prices (DataFrame) must be provided for anomaly detection"}

            result = await self._detect_anomalies(asset_historical_prices_df, threshold_param)
            return {"data": result}

        elif operation == "get_earnings_surprises":
            tickers = parameters.get("tickers")
            api_agent_instance = parameters.get("api_agent") # Orchestrator must pass this

            if not tickers:
                return {"error": "No tickers provided for get_earnings_surprises"}
            if not api_agent_instance:
                return {"error": "APIAgent instance not provided for get_earnings_surprises"}

            result = await self.get_earnings_surprises(tickers, api_agent_instance)
            return {"data": result} # The method already returns data in the desired structure

        else:
            return {"error": f"Unknown operation: {operation}"}

    async def _calculate_risk(
        self,
        asset_historical_prices: pd.DataFrame,
        benchmark_historical_prices: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """Calculate risk metrics for a single asset, optionally against a benchmark.

        Args:
            asset_historical_prices: DataFrame with 'Date' and 'Close' columns for the asset.
            benchmark_historical_prices: Optional DataFrame with 'Date' and 'Close' for benchmark.
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%).

        Returns:
            A dictionary of calculated risk metrics.
        """
        try:
            if not isinstance(asset_historical_prices, pd.DataFrame) or asset_historical_prices.empty:
                return {"error": "Asset historical prices must be a non-empty DataFrame."}
            if 'Date' not in asset_historical_prices.columns or 'Close' not in asset_historical_prices.columns:
                return {"error": "Asset historical prices DataFrame must contain 'Date' and 'Close' columns."}

            # Prepare asset data
            asset_df = asset_historical_prices.copy()
            asset_df['Date'] = pd.to_datetime(asset_df['Date'])
            asset_df = asset_df.set_index('Date').sort_index()
            asset_returns = asset_df['Close'].pct_change().dropna()

            if asset_returns.empty:
                return {"error": "Asset returns could not be calculated (e.g., single data point)."}

            # Volatility (Annualized)
            volatility = asset_returns.std() * np.sqrt(252)
            if pd.isna(volatility): volatility = 0.0 # Handle case with single return value (std is NaN)


            # Sharpe Ratio (Annualized)
            annualized_asset_return = asset_returns.mean() * 252
            if volatility == 0:
                sharpe_ratio = np.nan # Or some other indicator of undefined Sharpe
            else:
                sharpe_ratio = (annualized_asset_return - risk_free_rate) / volatility
            
            # Max Drawdown
            cumulative_returns = (1 + asset_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            if pd.isna(max_drawdown): max_drawdown = 0.0

            # Value at Risk (VaR) 95% (Historical)
            var_95 = np.percentile(asset_returns.dropna(), 5)

            beta = np.nan # Default to NaN if no benchmark

            if benchmark_historical_prices is not None and not benchmark_historical_prices.empty:
                if 'Date' not in benchmark_historical_prices.columns or 'Close' not in benchmark_historical_prices.columns:
                    logger.warning("Benchmark historical prices DataFrame is missing 'Date' or 'Close' columns. Skipping Beta calculation.")
                else:
                    benchmark_df = benchmark_historical_prices.copy()
                    benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
                    benchmark_df = benchmark_df.set_index('Date').sort_index()
                    benchmark_returns = benchmark_df['Close'].pct_change().dropna()

                    if not benchmark_returns.empty:
                        # Align data by merging on date index
                        aligned_df = pd.merge(asset_returns.rename("asset"), benchmark_returns.rename("benchmark"), left_index=True, right_index=True, how='inner')
                        
                        if len(aligned_df) > 1: # Need at least 2 data points for covariance
                            aligned_asset_returns = aligned_df["asset"]
                            aligned_benchmark_returns = aligned_df["benchmark"]

                            covariance_matrix = np.cov(aligned_asset_returns, aligned_benchmark_returns)
                            covariance_with_benchmark = covariance_matrix[0, 1]
                            variance_benchmark = np.var(aligned_benchmark_returns) # Using numpy's var for consistency

                            if variance_benchmark == 0:
                                beta = np.nan # Or some other indicator of undefined Beta
                            else:
                                beta = covariance_with_benchmark / variance_benchmark
                        else:
                            logger.warning("Not enough overlapping data points between asset and benchmark to calculate Beta.")
                    else:
                        logger.warning("Benchmark returns could not be calculated. Skipping Beta calculation.")

            result = {
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else None,
                "max_drawdown": float(max_drawdown),
                "value_at_risk_95": float(var_95),
                "beta": float(beta) if pd.notna(beta) else None,
            }
            logger.info(f"Calculated risk metrics: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calculating risk: {str(e)}")
            return {"error": f"Error calculating risk: {str(e)}"}

    async def _analyze_portfolio(
    self,
    portfolio: List[Dict[str, Any]], 
    portfolio_asset_historical_prices: pd.DataFrame,  # Moved up (required parameter)
    asset_metadata: Optional[Dict[str, Dict[str, Any]]] = None,  # Moved down (optional)
    benchmark_historical_prices: Optional[pd.DataFrame] = None, 
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
        """Analyze a portfolio's composition and risk.
        Args:
            portfolio: List of portfolio items with ticker and weight.
                       Example: [{"ticker": "AAPL", "weight": 0.25}, ...]
            asset_metadata: Dict of {"TICKER": stock_info_dict} for composition.
            portfolio_asset_historical_prices: DataFrame with 'Date' index and columns for each ticker's 'Close' price.
            benchmark_historical_prices: Optional DataFrame with 'Date' index and 'Close' column for benchmark.
            risk_free_rate: Annual risk-free rate.
        Returns:
            Portfolio analysis including composition and risk metrics.
        """
        try:
            # === Composition Analysis (uses asset_metadata) ===
            if not portfolio or not isinstance(portfolio, list):
                return {"error": "Invalid portfolio structure for composition analysis."}

            tickers = [item.get("ticker") for item in portfolio if item.get("ticker")]
            weights_list = [item.get("weight", 0) for item in portfolio if item.get("ticker")]

            if not tickers:
                 logger.warning("Portfolio composition analysis requested for an empty portfolio.")
                 # Initialize with empty/default values for the composition part
                 composition_result = {
                    "tickers": [], "weights": [], "sector_exposure": [], "country_exposure": [],
                    "asset_type_exposure": [], "concentration": 0, "diversification_status": "N/A"
                 }
            else:
                weight_sum = sum(weights_list)
                if abs(weight_sum - 1.0) > 1e-6 and weight_sum != 0:
                    logger.info(f"Normalizing portfolio weights for composition. Original sum: {weight_sum}")
                    weights_list = [w / weight_sum for w in weights_list]
                elif weight_sum == 0:
                    logger.warning("Portfolio weights sum to zero for composition. Distributing weights equally.")
                    weights_list = [1/len(tickers)] * len(tickers)
                
                sector_exposure, country_exposure, asset_type_exposure = {}, {}, {}
                for ticker, weight in zip(tickers, weights_list):
                    stock_info = asset_metadata.get(ticker) if asset_metadata else None
                    sector = stock_info.get("sector", "Unknown") if stock_info else "Unknown"
                    country = stock_info.get("country", "Unknown") if stock_info else "Unknown"
                    quote_type = stock_info.get("quoteType") if stock_info else None
                    if quote_type == "EQUITY": asset_type = "Equity"
                    elif quote_type in ["MUTUALFUND", "ETF", "CRYPTOCURRENCY", "CURRENCY", "FUTURE", "OPTION", "INDEX"]: asset_type = quote_type
                    else: asset_type = "Equity" if stock_info else "Unknown"
                    if not stock_info: logger.warning(f"No asset metadata for {ticker} in composition analysis.")
                    
                    sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
                    country_exposure[country] = country_exposure.get(country, 0) + weight
                    asset_type_exposure[asset_type] = asset_type_exposure.get(asset_type, 0) + weight

                concentration = sum([w * w for w in weights_list])
                diversification_status = "Well Diversified" if concentration < 0.15 else ("Moderately Diversified" if concentration < 0.25 else "Concentrated")
                
                composition_result = {
                    "tickers": tickers, "weights": weights_list,
                    "sector_exposure": [{"sector": s, "weight": w} for s, w in sector_exposure.items()],
                    "country_exposure": [{"country": c, "weight": w} for c, w in country_exposure.items()],
                    "asset_type_exposure": [{"asset_type": at, "weight": w} for at, w in asset_type_exposure.items()],
                    "concentration": concentration, "diversification_status": diversification_status,
                }

            # === Risk Analysis (uses portfolio_asset_historical_prices) ===
            if not isinstance(portfolio_asset_historical_prices, pd.DataFrame) or portfolio_asset_historical_prices.empty:
                logger.warning("Portfolio historical prices DataFrame is missing or empty. Skipping portfolio risk metric calculations.")
                composition_result["risk_metrics"] = {"error": "Historical price data for portfolio assets not provided or empty."}
                return composition_result # Return composition even if risk cannot be calculated

            # Ensure 'Date' index is datetime
            if not isinstance(portfolio_asset_historical_prices.index, pd.DatetimeIndex):
                 try: # If Date is a column
                    portfolio_asset_historical_prices['Date'] = pd.to_datetime(portfolio_asset_historical_prices['Date'])
                    portfolio_asset_historical_prices = portfolio_asset_historical_prices.set_index('Date')
                 except KeyError: # Date column does not exist
                    return {"error": "Portfolio historical prices DataFrame must have a 'Date' index or 'Date' column."}

            portfolio_asset_historical_prices = portfolio_asset_historical_prices.sort_index()
            asset_returns_df = portfolio_asset_historical_prices.pct_change().dropna()

            if asset_returns_df.empty:
                logger.warning("Asset returns DataFrame is empty after processing. Skipping portfolio risk metrics.")
                composition_result["risk_metrics"] = {"error": "Could not calculate asset returns for portfolio risk analysis."}
                return composition_result

            # Create weights Series aligned with asset_returns_df columns
            # Ensure weights_list corresponds to the order of tickers found in asset_returns_df.columns
            # This assumes portfolio_asset_historical_prices columns are the tickers
            aligned_weights_list = []
            valid_tickers_for_returns = []
            portfolio_tickers_set = set(tickers) # Tickers from the portfolio composition

            for ticker_col in asset_returns_df.columns:
                if ticker_col in portfolio_tickers_set:
                    try:
                        idx = tickers.index(ticker_col)
                        aligned_weights_list.append(weights_list[idx])
                        valid_tickers_for_returns.append(ticker_col)
                    except ValueError: # Should not happen if logic is correct
                        logger.error(f"Ticker {ticker_col} from price data not found in portfolio composition tickers. This is unexpected.")
                        aligned_weights_list.append(0) # Assign zero weight if mismatch
                else: # Ticker in price data but not in portfolio definition (e.g. if price data is broader)
                    logger.warning(f"Ticker {ticker_col} in historical prices not in defined portfolio. Assigning zero weight for risk calculation.")
                    aligned_weights_list.append(0)


            if not valid_tickers_for_returns or sum(aligned_weights_list) == 0 :
                 logger.warning("No valid tickers with weights found matching historical price data columns. Cannot calculate portfolio returns.")
                 composition_result["risk_metrics"] = {"error": "No matching tickers between portfolio definition and historical price data columns for risk calculation."}
                 return composition_result

            # Normalize aligned_weights_list if it was modified (e.g. due to missing tickers)
            current_aligned_weight_sum = sum(aligned_weights_list)
            if abs(current_aligned_weight_sum - 1.0) > 1e-6 and current_aligned_weight_sum != 0:
                 logger.info(f"Normalizing aligned weights for risk calculation. Original sum: {current_aligned_weight_sum}")
                 aligned_weights_list = [w / current_aligned_weight_sum for w in aligned_weights_list]
            
            weights_series = pd.Series(aligned_weights_list, index=valid_tickers_for_returns)
            
            # Filter asset_returns_df to only include valid_tickers_for_returns to match weights_series
            filtered_asset_returns_df = asset_returns_df[valid_tickers_for_returns]

            portfolio_daily_returns = (filtered_asset_returns_df * weights_series).sum(axis=1)

            if portfolio_daily_returns.empty:
                logger.warning("Portfolio daily returns are empty. Skipping portfolio risk metrics.")
                composition_result["risk_metrics"] = {"error": "Portfolio daily returns could not be calculated."}
                return composition_result

            portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
            if pd.isna(portfolio_volatility): portfolio_volatility = 0.0

            annualized_portfolio_return = portfolio_daily_returns.mean() * 252
            portfolio_sharpe_ratio = (annualized_portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else np.nan
            
            cumulative_portfolio_returns = (1 + portfolio_daily_returns).cumprod()
            portfolio_peak = cumulative_portfolio_returns.cummax()
            portfolio_drawdown = (cumulative_portfolio_returns - portfolio_peak) / portfolio_peak
            portfolio_max_drawdown = portfolio_drawdown.min()
            if pd.isna(portfolio_max_drawdown): portfolio_max_drawdown = 0.0

            portfolio_var_95 = np.percentile(portfolio_daily_returns.dropna(), 5)
            portfolio_beta = np.nan

            if benchmark_historical_prices is not None and not benchmark_historical_prices.empty:
                if not isinstance(benchmark_historical_prices.index, pd.DatetimeIndex):
                    try:
                        benchmark_historical_prices['Date'] = pd.to_datetime(benchmark_historical_prices['Date'])
                        benchmark_historical_prices = benchmark_historical_prices.set_index('Date')
                    except KeyError:
                         logger.warning("Benchmark historical prices DataFrame must have a 'Date' index or 'Date' column. Skipping Beta.")
                    
                if isinstance(benchmark_historical_prices.index, pd.DatetimeIndex): # Check again if conversion succeeded
                    benchmark_historical_prices = benchmark_historical_prices.sort_index()
                    benchmark_returns = benchmark_historical_prices['Close'].pct_change().dropna()
                    if not benchmark_returns.empty:
                        aligned_returns = pd.merge(portfolio_daily_returns.rename("portfolio"), benchmark_returns.rename("benchmark"), left_index=True, right_index=True, how="inner")
                        if len(aligned_returns) > 1:
                            cov_matrix = np.cov(aligned_returns["portfolio"], aligned_returns["benchmark"])
                            benchmark_variance = np.var(aligned_returns["benchmark"])
                            portfolio_beta = cov_matrix[0, 1] / benchmark_variance if benchmark_variance != 0 else np.nan
                        else: logger.warning("Not enough overlapping data for portfolio beta calculation.")
                    else: logger.warning("Benchmark returns empty, skipping portfolio beta.")
                else: logger.warning("Benchmark Date index not DatetimeIndex after attempted conversion, skipping Beta.")


            composition_result["risk_metrics"] = {
                "portfolio_volatility": float(portfolio_volatility),
                "portfolio_sharpe_ratio": float(portfolio_sharpe_ratio) if pd.notna(portfolio_sharpe_ratio) else None,
                "portfolio_max_drawdown": float(portfolio_max_drawdown),
                "portfolio_value_at_risk_95": float(portfolio_var_95),
                "portfolio_beta": float(portfolio_beta) if pd.notna(portfolio_beta) else None,
            }
            
            logger.info(f"Analyzed portfolio '{tickers}'. Composition and risk metrics calculated.")
            return composition_result

        except Exception as e:
            logger.exception(f"Error analyzing portfolio: {str(e)}") # Use logger.exception for stack trace
            return {"error": f"Error in _analyze_portfolio: {str(e)}"}

    async def _analyze_performance(
        self,
        asset_historical_prices: pd.DataFrame,
        benchmark_historical_prices: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """Analyze performance of an asset, optionally against a benchmark.

        Args:
            asset_historical_prices: DataFrame with 'Date' and 'Close' columns for the asset.
            benchmark_historical_prices: Optional DataFrame with 'Date' and 'Close' for benchmark.
            risk_free_rate: Annual risk-free rate.

        Returns:
            A dictionary of calculated performance metrics.
        """
        try:
            if not isinstance(asset_historical_prices, pd.DataFrame) or asset_historical_prices.empty:
                return {"error": "Asset historical prices must be a non-empty DataFrame."}
            if 'Date' not in asset_historical_prices.columns or 'Close' not in asset_historical_prices.columns:
                return {"error": "Asset historical prices DataFrame must contain 'Date' and 'Close' columns."}

            # Prepare asset data
            asset_df = asset_historical_prices.copy()
            asset_df['Date'] = pd.to_datetime(asset_df['Date'])
            asset_df = asset_df.set_index('Date').sort_index()
            asset_returns = asset_df['Close'].pct_change().dropna()
            
            if asset_returns.empty:
                return {"error": "Asset returns could not be calculated (e.g., single data point)."}

            # Total Return
            if len(asset_df['Close']) < 2:
                return {"error": "Cannot calculate total return with less than 2 price points."}
            total_return = (asset_df['Close'].iloc[-1] / asset_df['Close'].iloc[0]) - 1

            # Annualized Return
            num_days = (asset_df.index[-1] - asset_df.index[0]).days
            num_years = num_days / 365.25 # More precise than using 252 for returns over multiple years
            if num_years > 0:
                annualized_return = (1 + total_return) ** (1 / num_years) - 1
            elif len(asset_returns) > 0 : # If less than a year but has returns
                 annualized_return = asset_returns.mean() * 252 # Fallback to simple annualization for short periods
            else:
                 annualized_return = 0.0 # Or total_return if preferred for very short periods

            # Volatility (Annualized)
            volatility = asset_returns.std() * np.sqrt(252)
            if pd.isna(volatility): volatility = 0.0

            # Sharpe Ratio (Annualized)
            # Use the more precise annualized_return if available, else simple mean for Sharpe
            # For consistency with common Sharpe Ratio definitions, often daily excess return is used.
            # Let's use mean daily return * 252 for annualized asset return in Sharpe.
            annualized_asset_return_for_sharpe = asset_returns.mean() * 252
            if volatility == 0:
                sharpe_ratio = np.nan
            else:
                sharpe_ratio = (annualized_asset_return_for_sharpe - risk_free_rate) / volatility

            # Max Drawdown
            cumulative_returns = (1 + asset_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            if pd.isna(max_drawdown): max_drawdown = 0.0

            # Monthly Returns
            monthly_returns_series = asset_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_list = [
                {"month": idx.strftime('%Y-%m'), "return": val}
                for idx, val in monthly_returns_series.items()
            ]


            result = {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else None,
                "max_drawdown": float(max_drawdown),
                "monthly_returns": monthly_returns_list,
                "alpha": None, # Default to None
                "beta": None,  # Default to None
                "tracking_error": None, # Default to None
                "information_ratio": None # Default to None
            }
            
            # Benchmark-related metrics
            if benchmark_historical_prices is not None and not benchmark_historical_prices.empty:
                if 'Date' not in benchmark_historical_prices.columns or 'Close' not in benchmark_historical_prices.columns:
                    logger.warning("Benchmark historical prices DataFrame is missing 'Date' or 'Close' columns. Skipping benchmark metrics.")
                else:
                    benchmark_df = benchmark_historical_prices.copy()
                    benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
                    benchmark_df = benchmark_df.set_index('Date').sort_index()
                    benchmark_returns = benchmark_df['Close'].pct_change().dropna()

                    if not benchmark_returns.empty:
                        # Align data
                        aligned_df = pd.merge(asset_returns.rename("asset"), benchmark_returns.rename("benchmark"), left_index=True, right_index=True, how='inner')
                        
                        if len(aligned_df) > 1:
                            aligned_asset_returns = aligned_df["asset"]
                            aligned_benchmark_returns = aligned_df["benchmark"]

                            # Beta
                            covariance_matrix = np.cov(aligned_asset_returns, aligned_benchmark_returns)
                            covariance_with_benchmark = covariance_matrix[0, 1]
                            variance_benchmark = np.var(aligned_benchmark_returns)
                            beta = covariance_with_benchmark / variance_benchmark if variance_benchmark != 0 else np.nan
                            result["beta"] = float(beta) if pd.notna(beta) else None

                            # Alpha (CAPM Alpha)
                            annualized_benchmark_return = aligned_benchmark_returns.mean() * 252
                            # Use the annualized_return calculated earlier for the asset
                            if pd.notna(beta):
                                alpha = annualized_return - (risk_free_rate + beta * (annualized_benchmark_return - risk_free_rate))
                                result["alpha"] = float(alpha) if pd.notna(alpha) else None
                            
                            # Tracking Error
                            active_returns = aligned_asset_returns - aligned_benchmark_returns
                            tracking_error = active_returns.std() * np.sqrt(252)
                            result["tracking_error"] = float(tracking_error) if pd.notna(tracking_error) else None

                            # Information Ratio
                            if tracking_error != 0 and pd.notna(tracking_error):
                                mean_active_return = active_returns.mean() * 252
                                information_ratio = mean_active_return / tracking_error
                                result["information_ratio"] = float(information_ratio) if pd.notna(information_ratio) else None
                        else:
                            logger.warning("Not enough overlapping data points between asset and benchmark for performance comparison.")
                    else:
                        logger.warning("Benchmark returns could not be calculated. Skipping benchmark metrics.")
            
            logger.info(f"Analyzed performance. Total Return: {result['total_return']:.2%}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {"error": f"Error analyzing performance: {str(e)}"}

    async def _detect_anomalies(
        self, asset_historical_prices: pd.DataFrame, threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in financial data using Z-score.

        Args:
            asset_historical_prices: DataFrame with 'Date' and 'Close' columns for the asset.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            List of detected anomalies, or a list containing an error dictionary.
        """
        try:
            if not isinstance(asset_historical_prices, pd.DataFrame) or asset_historical_prices.empty:
                return [{"error": "Asset historical prices must be a non-empty DataFrame."}]
            
            # Determine if 'Date' is index or column
            df = asset_historical_prices.copy()
            date_is_index = isinstance(df.index, pd.DatetimeIndex)

            if not date_is_index and 'Date' not in df.columns:
                 return [{"error": "Asset historical prices DataFrame must contain 'Date' column or have a DatetimeIndex."}]
            if not date_is_index:
                df['Date'] = pd.to_datetime(df['Date'])
                # df = df.set_index('Date') # Not strictly necessary if we handle both cases for output

            if 'Close' not in df.columns:
                return [{"error": "Asset historical prices DataFrame must contain 'Close' column."}]

            df_returns = df['Close'].pct_change().dropna()

            if df_returns.empty:
                logger.info("No returns to analyze for anomalies (e.g., single data point or all NaNs).")
                return []

            mean_return = df_returns.mean()
            std_return = df_returns.std()

            if pd.isna(std_return) or std_return == 0:
                logger.info("Standard deviation of returns is NaN or zero. Cannot calculate Z-scores for anomalies.")
                return []

            # Calculate Z-scores on the original DataFrame to keep all columns
            # We need to align the returns with the original DataFrame for this
            df['Return'] = df['Close'].pct_change() # Keep NaNs for non-return rows
            df['Z_Score'] = (df['Return'] - mean_return) / std_return
            
            # Detect anomalies based on z-score threshold
            # Ensure we only consider rows where Z_Score is not NaN
            anomalies_df = df[df['Z_Score'].abs() > threshold].copy()

            anomalies_list = []
            for index, row in anomalies_df.iterrows():
                # Handle date extraction based on whether it was originally index or column
                date_value = None
                if date_is_index:
                    date_value = index.isoformat() if pd.notna(index) else None
                elif 'Date' in row and pd.notna(row['Date']):
                    date_value = pd.to_datetime(row['Date']).isoformat()


                anomaly = {
                    "date": date_value,
                    "close": float(row["Close"]),
                    "return": float(row["Return"]) if pd.notna(row["Return"]) else None,
                    "z_score": float(row["Z_Score"]) if pd.notna(row["Z_Score"]) else None,
                    "direction": "up" if row["Return"] > 0 else ("down" if row["Return"] < 0 else "flat"),
                    "magnitude": ("extreme" if abs(row["Z_Score"]) > 3.0 else "significant") if pd.notna(row["Z_Score"]) else None,
                }
                anomalies_list.append(anomaly)

            logger.info(f"Detected {len(anomalies_list)} anomalies with threshold {threshold}")
            return anomalies_list

        except Exception as e:
            logger.exception(f"Error detecting anomalies: {str(e)}") # Use logger.exception for stack trace
            return [{"error": f"Error detecting anomalies: {str(e)}"}]

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
    ) -> Dict[str, Any]: # This method actually returns List[Dict[str,Any]] or List with error dict
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
        # This public method's return type should align with what self.run(request) returns
        # which is typically a Dict[str, Any] containing either "data" or "error".
        # The _detect_anomalies method returns List[Dict[str,Any]] under "data" key if successful.
        return await self.run(request) # This will be Dict[str, Any]

    async def get_earnings_surprises(
        self, tickers: List[str], api_agent: Any # Using Any to avoid direct APIAgent import for now
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate earnings surprises for a list of tickers.

        Args:
            tickers: A list of stock ticker symbols.
            api_agent: An instance of APIAgent (or compatible) to fetch earnings history.

        Returns:
            A dictionary where keys are tickers and values are lists of earnings surprise data.
            Example:
            {
                "AAPL": [
                    {
                        "report_date": "YYYY-MM-DD",
                        "fiscal_period": "Qx YYYY", // Or date if fiscal period not available
                        "estimated_eps": 1.50,
                        "reported_eps": 1.55,
                        "surprise_eps": 0.05,
                        "surprise_percentage": 3.33
                    }, ...
                ]
            }
        """
        all_surprises: Dict[str, List[Dict[str, Any]]] = {}
        
        if not api_agent or not hasattr(api_agent, 'get_earnings_history'):
            logger.error("APIAgent instance is invalid or missing 'get_earnings_history' method.")
            return {"error": "Invalid APIAgent provided."} # type: ignore

        for ticker in tickers:
            try:
                logger.debug(f"Fetching earnings history for {ticker}")
                # APIAgent.get_earnings_history returns a Dict: {"data": list_of_dicts | error_dict}
                earnings_history_response = await api_agent.get_earnings_history(ticker)
                
                # Check for errors from APIAgent
                if "error" in earnings_history_response:
                    logger.warning(f"Could not fetch earnings history for {ticker}: {earnings_history_response['error']}")
                    all_surprises[ticker] = [{"error": f"Failed to fetch earnings history: {earnings_history_response['error']}"}]
                    continue

                earnings_data_list = earnings_history_response.get("data")

                if not earnings_data_list or not isinstance(earnings_data_list, list):
                    logger.info(f"No earnings data or unexpected format for {ticker}: {earnings_data_list}")
                    all_surprises[ticker] = [] # No data or empty
                    continue
                
                # Convert list of dicts to DataFrame
                # APIAgent should already provide dates as strings in 'YYYY-MM-DD' format in the 'Earnings Date' column (or similar)
                # The columns from yfinance.earnings_dates are typically 'EPS Estimate', 'Reported EPS', and the index is 'Earnings Date'
                earnings_df = pd.DataFrame(earnings_data_list)

                # Ensure necessary columns are present
                # yfinance might use slightly different names, e.g. 'Earnings Date' is the index.
                # The APIAgent already processes this into a column. Let's assume the column from APIAgent is 'Earnings Date'
                date_col_options = ['Earnings Date', 'Report Date', 'index'] # Common names after reset_index
                date_col = next((col for col in date_col_options if col in earnings_df.columns), None)

                if not date_col:
                    logger.warning(f"Earnings date column not found in data for {ticker}. Columns: {earnings_df.columns}")
                    all_surprises[ticker] = [{"error": "Earnings date column not found."}]
                    continue

                required_cols = ['EPS Estimate', 'Reported EPS']
                if not all(col in earnings_df.columns for col in required_cols):
                    logger.warning(f"Missing required EPS columns in earnings data for {ticker}. Found: {earnings_df.columns}")
                    all_surprises[ticker] = [{"error": "Missing required EPS columns."}]
                    continue

                # Convert date column to datetime objects for sorting, if not already
                try:
                    earnings_df[date_col] = pd.to_datetime(earnings_df[date_col])
                except Exception as date_parse_err:
                    logger.warning(f"Could not parse dates for {ticker}: {date_parse_err}. Dates: {earnings_df[date_col].head()}")
                    all_surprises[ticker] = [{"error": f"Could not parse dates: {date_parse_err}"}]
                    continue

                # Sort by date to get recent surprises (descending)
                earnings_df = earnings_df.sort_values(by=date_col, ascending=False)

                # Drop rows with missing essential data
                # yfinance sometimes uses '-' for missing values, which might become NaN or stay as string.
                # Convert to numeric, coercing errors to NaN
                earnings_df['EPS Estimate'] = pd.to_numeric(earnings_df['EPS Estimate'], errors='coerce')
                earnings_df['Reported EPS'] = pd.to_numeric(earnings_df['Reported EPS'], errors='coerce')
                
                valid_earnings_df = earnings_df.dropna(subset=['EPS Estimate', 'Reported EPS'])

                if valid_earnings_df.empty:
                    logger.info(f"No valid earnings rows with both estimate and reported EPS for {ticker}.")
                    all_surprises[ticker] = []
                    continue

                ticker_surprises = []
                for _, row in valid_earnings_df.head(8).iterrows(): # Take last 8 quarters (after sorting)
                    estimated_eps = row['EPS Estimate']
                    reported_eps = row['Reported EPS']
                    
                    surprise_eps = reported_eps - estimated_eps
                    
                    if estimated_eps != 0:
                        surprise_percentage = (surprise_eps / abs(estimated_eps)) * 100
                    elif surprise_eps == 0: # Both estimate and actual are 0
                        surprise_percentage = 0.0
                    else: # Estimate was 0, but actual was not. Assign a large percentage or handle as special case.
                        surprise_percentage = float('inf') if surprise_eps > 0 else float('-inf')


                    ticker_surprises.append({
                        "report_date": row[date_col].strftime('%Y-%m-%d'),
                        "fiscal_period": f"{row[date_col].year} Q{((row[date_col].month - 1) // 3) + 1}", # Approximate fiscal period
                        "estimated_eps": estimated_eps,
                        "reported_eps": reported_eps,
                        "surprise_eps": round(surprise_eps, 4),
                        "surprise_percentage": round(surprise_percentage, 2) if pd.notna(surprise_percentage) and np.isfinite(surprise_percentage) else "N/A"
                    })
                
                all_surprises[ticker] = ticker_surprises
                logger.info(f"Calculated {len(ticker_surprises)} earnings surprises for {ticker}")

            except Exception as e:
                logger.exception(f"Error calculating earnings surprises for {ticker}: {str(e)}")
                all_surprises[ticker] = [{"error": f"An unexpected error occurred: {str(e)}"}]
        
        return all_surprises

    async def fetch_earnings_surprises(
        self, tickers: List[str], api_agent: Any # Using Any to avoid direct APIAgent import for now
    ) -> Dict[str, Any]:
        """Public convenience method to fetch earnings surprises.

        Args:
            tickers: A list of stock ticker symbols.
            api_agent: An instance of APIAgent.

        Returns:
            A dictionary containing earnings surprise data or an error.
        """
        request = {
            "operation": "get_earnings_surprises",
            "parameters": {"tickers": tickers, "api_agent": api_agent},
        }
        return await self.run(request)
