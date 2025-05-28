# API Reference

This document provides a comprehensive reference for the Finance Agent API endpoints.

## Base URL

When running locally: `http://localhost:8000`

## Authentication

Current implementation does not require authentication. Future versions will implement API key authentication.

## Endpoints

### Market Data

#### Get Stock Information


```http
GET /api/v1/market/stock/{ticker}
```


Returns current information about a specific stock.

**Parameters:**

- `ticker` (path parameter): The stock ticker symbol (e.g., AAPL, MSFT, GOOGL)

**Response:**

```json
{
  "data": {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "price": 180.95,
    "change": 1.25,
    "change_percent": 0.69,
    "market_cap": "2.85T",
    "pe_ratio": 29.76,
    "dividend_yield": 0.51,
    "sector": "Technology",
    "timestamp": "2025-05-28T14:30:00"
  },
  "agent": "API Agent",
  "processing_time": 0.125,
  "timestamp": "2025-05-28T14:30:05"
}
```

#### Get Market Summary

```http
GET /api/v1/market/summary
```

Returns a summary of major market indices.

**Response:**

```json
{
  "data": {
    "^GSPC": {
      "name": "S&P 500",
      "price": 4984.35,
      "change": 12.75,
      "change_percent": 0.25
    },
    "^DJI": {
      "name": "Dow Jones Industrial Average",
      "price": 38456.78,
      "change": -34.56,
      "change_percent": -0.09
    },
    "^IXIC": {
      "name": "NASDAQ Composite",
      "price": 16789.54,
      "change": 78.45,
      "change_percent": 0.47
    },
    "timestamp": "2025-05-28T14:30:00"
  },
  "agent": "API Agent",
  "processing_time": 0.152,
  "timestamp": "2025-05-28T14:30:05"
}
```

#### Get Historical Data

```
GET /api/v1/market/history/{ticker}
```

Returns historical price data for a specific stock.

**Parameters:**

- `ticker` (path parameter): The stock ticker symbol
- `period` (query parameter): The time period (default: 1mo, options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
- `interval` (query parameter): The data interval (default: 1d, options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

**Response:**

```json
{
  "data": {
    "symbol": "AAPL",
    "period": "1mo",
    "interval": "1d",
    "prices": [
      {"date": "2025-04-28", "open": 175.23, "high": 176.45, "low": 174.56, "close": 175.67, "volume": 52364789},
      {"date": "2025-04-29", "open": 175.89, "high": 177.12, "low": 175.23, "close": 176.78, "volume": 48756321},
      // More data points...
      {"date": "2025-05-28", "open": 179.67, "high": 181.23, "low": 179.45, "close": 180.95, "volume": 51234567}
    ]
  },
  "agent": "API Agent",
  "processing_time": 0.318,
  "timestamp": "2025-05-28T14:30:10"
}
```

### News and Analysis

#### Get Financial News

```
GET /api/v1/news
```

Returns recent financial news articles.

**Parameters:**

- `keywords` (query parameter, optional): Filter news by keywords
- `sources` (query parameter, optional): Filter by news sources
- `limit` (query parameter, optional): Number of articles to return (default: 10, max: 50)

**Response:**

```json
{
  "data": {
    "articles": [
      {
        "title": "Fed Signals Potential Rate Cut in September Meeting",
        "source": "Bloomberg",
        "url": "https://bloomberg.com/news/articles/...",
        "published_at": "2025-05-28T10:15:00",
        "summary": "Federal Reserve officials signaled they may cut interest rates..."
      },
      // More articles...
    ],
    "count": 10,
    "total": 45
  },
  "agent": "Scraping Agent",
  "processing_time": 0.876,
  "timestamp": "2025-05-28T14:30:15"
}
```

#### Get SEC Filings

```
GET /api/v1/filings/{ticker}
```

Returns recent SEC filings for a specific company.

**Parameters:**

- `ticker` (path parameter): The stock ticker symbol
- `filing_type` (query parameter, optional): Filter by filing type (e.g., 10-K, 10-Q, 8-K)
- `limit` (query parameter, optional): Number of filings to return (default: 5, max: 20)

**Response:**

```json
{
  "data": {
    "company": "Apple Inc.",
    "symbol": "AAPL",
    "filings": [
      {
        "type": "10-Q",
        "title": "Quarterly Report",
        "filed_at": "2025-04-30",
        "url": "https://www.sec.gov/Archives/edgar/data/...",
        "description": "Quarterly report for the period ending March 31, 2025"
      },
      // More filings...
    ]
  },
  "agent": "Scraping Agent",
  "processing_time": 0.654,
  "timestamp": "2025-05-28T14:30:20"
}
```

### Portfolio and Analysis

#### Analyze Portfolio

```
POST /api/v1/portfolio/analyze
```

Analyzes a portfolio of stocks and provides performance metrics and risk analysis.

**Request Body:**

```json
{
  "portfolio": [
    {"ticker": "AAPL", "weight": 0.25, "shares": 100},
    {"ticker": "MSFT", "weight": 0.25, "shares": 50},
    {"ticker": "GOOGL", "weight": 0.20, "shares": 30},
    {"ticker": "AMZN", "weight": 0.15, "shares": 25},
    {"ticker": "NVDA", "weight": 0.15, "shares": 40}
  ],
  "benchmark": "SPY",
  "period": "1y"
}
```

**Response:**

```json
{
  "data": {
    "portfolio_value": 156789.45,
    "performance": {
      "return_1m": 2.34,
      "return_3m": 5.67,
      "return_6m": 8.91,
      "return_1y": 15.23,
      "annualized_return": 15.23,
      "benchmark_return_1y": 12.45
    },
    "risk_metrics": {
      "volatility": 18.72,
      "sharpe_ratio": 0.81,
      "sortino_ratio": 1.24,
      "max_drawdown": -12.34,
      "beta": 1.15,
      "alpha": 2.78
    },
    "sector_allocation": {
      "Technology": 75.0,
      "Consumer Discretionary": 15.0,
      "Communication Services": 10.0
    },
    "recommendations": [
      "Your portfolio is heavily weighted towards technology stocks, consider diversifying into other sectors.",
      "The high beta of 1.15 indicates more volatility than the market."
    ]
  },
  "agent": "Analysis Agent",
  "processing_time": 1.235,
  "timestamp": "2025-05-28T14:30:30"
}
```

### Natural Language Queries

#### Process Query

```
POST /api/v1/query
```

Processes a natural language query and returns a relevant response.

**Request Body:**

```json
{
  "query": "What is the current performance of Apple stock and how does it compare to the market?",
  "use_voice": false
}
```

**Response:**

```json
{
  "data": {
    "query": "What is the current performance of Apple stock and how does it compare to the market?",
    "response": "Apple (AAPL) is currently trading at $180.95, up 0.69% today. Over the past month, Apple has gained 3.2% compared to the S&P 500's 2.1% increase. The stock has outperformed the broader market with a year-to-date return of 22.4% versus the S&P 500's 15.7%. The company recently reported strong quarterly earnings, beating analyst expectations with revenue of $94.8 billion and EPS of $1.52.",
    "sources": [
      {"type": "market_data", "description": "Current stock price and performance"},
      {"type": "news", "title": "Apple Reports Q2 2025 Earnings", "source": "CNBC"}
    ],
    "audio_url": null
  },
  "agent": "Orchestrator",
  "processing_time": 1.567,
  "timestamp": "2025-05-28T14:30:40"
}
```

#### Generate Market Brief

```
POST /api/v1/market/brief
```

Generates a comprehensive market brief with optional voice output.

**Request Body:**

```json
{
  "include_portfolio": true,
  "portfolio": [
    {"ticker": "AAPL", "weight": 0.25},
    {"ticker": "MSFT", "weight": 0.25},
    {"ticker": "GOOGL", "weight": 0.20},
    {"ticker": "AMZN", "weight": 0.15},
    {"ticker": "NVDA", "weight": 0.15}
  ],
  "use_voice": true
}
```

**Response:**

```json
{
  "data": {
    "brief": {
      "market_summary": "Good morning. Here's your market brief for Wednesday, May 28, 2025. U.S. stocks are mixed in early trading, with technology shares leading gains while energy stocks decline. The S&P 500 is up 0.25% at 4,984, while the Dow Jones Industrial Average is down slightly by 0.09% at 38,456. The Nasdaq Composite is showing strength, up 0.47% at 16,789.",
      "key_events": "Federal Reserve minutes released yesterday indicate officials are considering rate cuts in the coming months as inflation continues to moderate. In corporate news, Apple announced a new AI initiative that sent the stock up nearly 3% in pre-market trading.",
      "portfolio_update": "Your portfolio is up 0.38% today, outperforming the S&P 500 by 0.13%. Top performers include NVDA, up 2.1%, and AAPL, up 1.2%. AMZN is the only decliner, down 0.3%. Year-to-date, your portfolio has returned 18.7%, ahead of the S&P 500's 15.7%.",
      "outlook": "Market sentiment remains cautiously optimistic with expectations for Fed rate cuts providing support. However, upcoming inflation data could introduce volatility. The technology sector continues to show resilience, which should benefit your current portfolio allocation."
    },
    "audio_url": "/api/v1/audio/brief_20250528_081500.mp3"
  },
  "agent": "Orchestrator",
  "processing_time": 3.245,
  "timestamp": "2025-05-28T14:31:00"
}
```

### Voice Interactions

#### Text to Speech

```
POST /api/v1/voice/tts
```

Converts text to speech and returns an audio URL.

**Request Body:**

```json
{
  "text": "Apple stock is currently trading at $180.95, up 0.69% today.",
  "voice_id": "default"
}
```

**Response:**

```json
{
  "data": {
    "audio_url": "/api/v1/audio/tts_20250528_143115.mp3",
    "duration_seconds": 4.5
  },
  "agent": "Voice Agent",
  "processing_time": 0.987,
  "timestamp": "2025-05-28T14:31:15"
}
```

#### Speech to Text

```
POST /api/v1/voice/stt
```

Converts speech audio to text.

**Request Body:**

Multipart form data with an audio file.

**Response:**

```json
{
  "data": {
    "text": "What's the current price of Apple stock?",
    "confidence": 0.98
  },
  "agent": "Voice Agent",
  "processing_time": 1.234,
  "timestamp": "2025-05-28T14:31:30"
}
```

## Error Handling

All API endpoints follow a consistent error format:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "The requested resource was not found.",
    "details": "Stock ticker 'INVALID' does not exist."
  },
  "timestamp": "2025-05-28T14:32:00"
}
```

Common error codes:

- `BAD_REQUEST`: The request was invalid or cannot be served
- `UNAUTHORIZED`: Authentication is required or failed
- `FORBIDDEN`: The request is understood but refused
- `NOT_FOUND`: The requested resource does not exist
- `INTERNAL_ERROR`: An error occurred on the server
- `SERVICE_UNAVAILABLE`: The service is temporarily unavailable

## Rate Limiting

API requests are limited to 100 requests per minute per IP address. When the rate limit is exceeded, the API will return a 429 Too Many Requests response.

## Webhooks

Webhooks for real-time notifications will be available in a future version.
