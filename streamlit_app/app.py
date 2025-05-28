"""Streamlit application for the Finance Agent."""

import os
import io
import json
import base64
import requests
import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# Add the project root directory to Python path to find the config module
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set page configuration first - this must be the first Streamlit command
st.set_page_config(
    page_title="Finance Agent",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import Config

# Check configuration and prompt for missing API keys
if not Config.validate():
    st.warning("Missing required API keys. Please provide them to enable full functionality.")
    # Only prompt for keys in streamlit if they're still missing after potential CLI prompts
    missing_api_groups = []
    for group_name, keys in Config.REQUIRED_API_GROUPS.items():
        if not any(getattr(Config, key) for key in keys):
            missing_api_groups.append((group_name, keys))
    
    if missing_api_groups:
        st.error("Please provide at least one API key for each required service:")
        for group_name, keys in missing_api_groups:
            with st.expander(f"Configure {group_name.upper()} API Keys"):
                for key in keys:
                    api_key = st.text_input(f"Enter your {key}:", type="password", key=f"input_{key}")
                    if api_key and st.button(f"Save {key}", key=f"save_{key}"):
                        # Save to environment variables and update Config
                        os.environ[key] = api_key
                        setattr(Config, key, api_key)
                        # Save to .env file
                        Config._save_key_to_env_file(key, api_key)
                        st.success(f"{key} saved successfully! Please refresh the page.")
                        st.experimental_rerun()

# Define API endpoint
API_URL = f"http://{Config.FASTAPI_HOST}:{Config.FASTAPI_PORT}"

# Initialize session state for history tracking
if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Ask Question"

if "market_brief" not in st.session_state:
    st.session_state.market_brief = None

if "recording" not in st.session_state:
    st.session_state.recording = False

# App title and description
st.title("Finance Assistant")
st.markdown(
    """
    A multi-agent finance assistant providing market briefs and answering financial queries.
    Get updates on market trends, portfolio details, and risk exposure with voice interaction capabilities.
    """
)

# Helper functions
def get_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def audio_to_base64(audio_path):
    """Convert audio file to base64 for HTML embedding."""
    if not os.path.exists(audio_path):
        return None
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return audio_b64

def plot_portfolio_allocation(portfolio_data):
    """Create a pie chart of portfolio allocation."""
    if not portfolio_data or "sector_exposure" not in portfolio_data:
        # Return placeholder data
        sectors = ["Technology", "Healthcare", "Financial", "Consumer", "Energy", "Other"]
        values = [0.35, 0.20, 0.15, 0.12, 0.08, 0.10]
        df = pd.DataFrame({"Sector": sectors, "Allocation": values})
    else:
        sectors = [item["sector"] for item in portfolio_data["sector_exposure"]]
        values = [item["weight"] for item in portfolio_data["sector_exposure"]]
        df = pd.DataFrame({"Sector": sectors, "Allocation": values})
    
    fig = px.pie(df, values="Allocation", names="Sector", title="Portfolio Sector Allocation")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig

def plot_stock_performance(stock_data):
    """Create a line chart of stock performance."""
    # Generate sample data if real data is not available
    dates = pd.date_range(end=datetime.now(), periods=30, freq='B')
    values = np.cumsum(np.random.normal(0, 1, 30)) + 100
    
    df = pd.DataFrame({"Date": dates, "Value": values})
    fig = px.line(df, x="Date", y="Value", title="Portfolio Performance (Last 30 Days)")
    return fig

# Sidebar for API status and settings
with st.sidebar:
    st.header("Settings & Status")
    
    # API Status
    api_health = get_api_health()
    if api_health:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
    
    # Input/Output Settings
    st.subheader("Input/Output Settings")
    
    # Voice Input Toggle
    voice_input = st.toggle("Enable Voice Input", value=False)
    if voice_input and not st.session_state.get("recording"):
        st.session_state.recording = False
    
    # Output Preference
    st.write("Response Format:")
    output_preference = st.radio(
        "How would you like to receive responses?",
        options=["Text Only", "Voice (English)", "Both Text and Voice"],
        index=0,
        horizontal=True,
        key="output_preference"
    )
    
    # Store preference in session state
    st.session_state.output_preference = output_preference
    
    # Market brief schedule
    st.subheader("Market Brief")
    st.write(f"Scheduled daily at {Config.MARKET_BRIEF_HOUR:02d}:{Config.MARKET_BRIEF_MINUTE:02d} {Config.TIMEZONE}")
    
    if st.button("Generate Market Brief Now"):
        with st.spinner("Generating market brief..."):
            try:
                response = requests.post(f"{API_URL}/market-brief")
                if response.status_code == 200:
                    st.session_state.market_brief = response.json()
                    st.session_state.current_tab = "Market Brief"
                    st.success("Market brief generated successfully!")
                else:
                    st.error(f"Failed to generate market brief: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main content area with tabs
tabs = st.tabs(["Ask Question", "Market Brief", "Portfolio Analysis"])

# 1. Ask Question Tab
with tabs[0]:
    if st.session_state.current_tab == "Ask Question":
        st.header("Ask a Financial Question")
        
        # Voice input handling
        if voice_input:
            st.info("🎤 Click the microphone button and speak your question")
            col1, col2 = st.columns([1, 5])
            
            with col1:
                if not st.session_state.recording:
                    if st.button("🎤 Record"):
                        st.session_state.recording = True
                else:
                    if st.button("⏹️ Stop"):
                        st.session_state.recording = False
                        # Here we would process the audio recording
                        # For now, we'll simulate this with a text input
                        st.info("Audio processed! Voice transcription would happen here in a real implementation.")
            
            with col2:
                # Fallback text input
                query = st.text_input("Question (type if voice not working)", key="voice_query", 
                                    placeholder="What's our risk exposure in Asia tech stocks today?")
        else:
            # Regular text input
            query = st.text_input("Question", key="text_query", 
                                placeholder="What's our risk exposure in Asia tech stocks today?")
        
        # Process query when ask button is clicked
        if st.button("Ask", type="primary") or query:
            if query:
                with st.spinner("Processing your question..."):
                    try:
                        # Determine voice output based on user preference
                        voice_output = st.session_state.output_preference in ["Voice (English)", "Both Text and Voice"]
                        
                        # Create the request payload
                        payload = {
                            "query": query,
                            "voice_input": False,  # We're handling voice input separately
                            "voice_output": voice_output
                        }
                        
                        # Send the request to the API
                        response = requests.post(f"{API_URL}/query", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Save to history
                            st.session_state.query_history.append({
                                "query": query,
                                "response": result["response"],
                                "timestamp": datetime.now().isoformat(),
                                "audio_url": result.get("audio_url"),
                            })
                            
                            # Display the response
                            st.subheader("Answer")
                            st.write(result["response"])
                            
                            # Determine if voice output is enabled based on user preference
                            voice_output = st.session_state.output_preference in ["Voice (English)", "Both Text and Voice"]
                            
                            # Display audio if available and voice output is enabled
                            if voice_output and result.get("audio_url"):
                                audio_url = result["audio_url"]
                                if audio_url.startswith("file://"):
                                    # Convert local file URL to actual path
                                    audio_path = audio_url.replace("file://", "")
                                    audio_b64 = audio_to_base64(audio_path)
                                    if audio_b64:
                                        st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                                else:
                                    st.audio(audio_url, format="audio/mp3")
                            
                            # Display sources if available
                            if result.get("sources") and len(result["sources"]) > 0:
                                st.subheader("Sources")
                                for source in result["sources"]:
                                    st.markdown(f"- [{source.get('title', 'Source')}]({source.get('url', '#')})")
                            
                            # Display processing time
                            st.caption(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
                        
                        else:
                            st.error(f"Error processing query: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question")
        
        # Display query history
        if st.session_state.query_history:
            st.divider()
            st.subheader("Recent Questions")
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Q: {item['query']}", expanded=(i == 0)):
                    st.write(item["response"])
                    timestamp = datetime.fromisoformat(item["timestamp"])
                    st.caption(f"Asked on {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

# 2. Market Brief Tab
with tabs[1]:
    if st.session_state.current_tab == "Market Brief" or "Market Brief" in st.session_state.current_tab:
        st.header("Daily Market Brief")
        
        # Check if we have a market brief
        if st.session_state.market_brief:
            brief = st.session_state.market_brief
            
            # Display date and title
            st.subheader(f"{brief.get('title', 'Daily Market Brief')} - {brief.get('date', datetime.now().strftime('%Y-%m-%d'))}")
            
            # Display summary
            st.write(brief.get("summary", "No summary available."))
            
            # Play audio if available
            if brief.get("audio_url"):
                audio_url = brief["audio_url"]
                if audio_url.startswith("file://"):
                    # Convert local file URL to actual path
                    audio_path = audio_url.replace("file://", "")
                    audio_b64 = audio_to_base64(audio_path)
                    if audio_b64:
                        st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                else:
                    st.audio(audio_url, format="audio/mp3")
            
            # Display sections in tabs
            sections = brief.get("sections", {})
            if sections:
                section_tabs = st.tabs(["Market Overview", "Portfolio Performance", "News Highlights", "Outlook"])
                
                with section_tabs[0]:
                    st.write(sections.get("market_overview", "No market overview available."))
                
                with section_tabs[1]:
                    st.write(sections.get("portfolio_performance", "No portfolio performance available."))
                    
                    # Add a visualization
                    st.plotly_chart(plot_stock_performance(None))
                
                with section_tabs[2]:
                    st.write(sections.get("news_highlights", "No news highlights available."))
                
                with section_tabs[3]:
                    st.write(sections.get("outlook", "No outlook available."))
            
            # Full text in expander
            with st.expander("View Full Report"):
                st.markdown(brief.get("full_text", "Full report not available."))
        
        else:
            st.info("No market brief available yet. Click 'Generate Market Brief Now' in the sidebar to create one.")
            
            # Show placeholder data
            st.subheader("Sample Market Brief")
            st.write("Below is an example of what a market brief looks like:")
            
            st.markdown("""
            ## Market Overview
            Markets closed higher today, with technology stocks leading the rally. 
            The S&P 500 gained 0.8%, while the Nasdaq Composite surged 1.2%. 
            Treasury yields fell as investors positioned for potential rate cuts later this year.
            
            ## Portfolio Performance
            Your portfolio gained 0.9% today, outperforming the S&P 500 by 0.1%. 
            Technology holdings were the main contributors, with semiconductor positions adding 25 basis points to performance.
            """)
            
            # Add a sample visualization
            st.plotly_chart(plot_stock_performance(None))

# 3. Portfolio Analysis Tab
with tabs[2]:
    st.header("Portfolio Analysis")
    
    # Placeholder portfolio data
    sample_portfolio = [
        {"ticker": "AAPL", "weight": 0.15, "sector": "Technology"},
        {"ticker": "MSFT", "weight": 0.12, "sector": "Technology"},
        {"ticker": "AMZN", "weight": 0.10, "sector": "Consumer"},
        {"ticker": "GOOGL", "weight": 0.08, "sector": "Communication"},
        {"ticker": "JNJ", "weight": 0.06, "sector": "Healthcare"},
        {"ticker": "JPM", "weight": 0.05, "sector": "Financial"},
        {"ticker": "V", "weight": 0.05, "sector": "Financial"},
        {"ticker": "PG", "weight": 0.04, "sector": "Consumer"},
        {"ticker": "UNH", "weight": 0.04, "sector": "Healthcare"},
        {"ticker": "Others", "weight": 0.31, "sector": "Various"},
    ]
    
    # Portfolio visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample sector allocation chart
        sector_data = {"sector_exposure": []}
        sectors = {}
        for item in sample_portfolio:
            sector = item["sector"]
            weight = item["weight"]
            if sector in sectors:
                sectors[sector] += weight
            else:
                sectors[sector] = weight
        
        for sector, weight in sectors.items():
            sector_data["sector_exposure"].append({"sector": sector, "weight": weight})
        
        st.plotly_chart(plot_portfolio_allocation(sector_data))
    
    with col2:
        # Sample performance chart
        st.plotly_chart(plot_stock_performance(None))
    
    # Portfolio holdings table
    st.subheader("Portfolio Holdings")
    df_portfolio = pd.DataFrame(sample_portfolio)
    df_portfolio["weight"] = df_portfolio["weight"].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_portfolio, use_container_width=True)
    
    # Risk metrics
    st.subheader("Risk Metrics")
    risk_metrics = {
        "Volatility (Annualized)": "12.5%",
        "Sharpe Ratio": "1.2",
        "Maximum Drawdown": "15.8%",
        "Beta": "0.92",
        "Value at Risk (95%)": "2.1%"
    }
    
    df_risk = pd.DataFrame({
        "Metric": risk_metrics.keys(),
        "Value": risk_metrics.values()
    })
    st.dataframe(df_risk, use_container_width=True, hide_index=True)
    
    # Risk assessment
    st.subheader("Risk Assessment")
    st.write("""
    Your portfolio currently has a **moderate risk profile** with a beta of 0.92 relative to the S&P 500. 
    The technology sector represents your largest exposure at 27%, which is slightly overweight compared to the benchmark. 
    Your Asian technology exposure is approximately 12% of the total portfolio, with the largest positions in TSMC and Samsung Electronics.
    
    Based on recent market conditions, we recommend maintaining your current allocation but monitoring developments in the semiconductor industry closely, 
    as regulatory headwinds in this space could create volatility in the coming months.
    """)

# Set the active tab based on session state
if st.session_state.current_tab == "Market Brief":
    st.session_state.current_tab = "" # Reset to avoid re-triggering
    st.rerun()

# Footer
st.markdown("---")
st.caption(f"© {datetime.now().year} Finance Agent | Built with Streamlit | Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Sidebar for settings and market brief
with st.sidebar:
    st.header("Settings")
    voice_input = st.toggle("Voice Input", value=False)
    voice_output = st.toggle("Voice Output", value=True)
    
    st.header("Daily Market Brief")
    st.write(f"Scheduled daily at {Config.MARKET_BRIEF_HOUR:02d}:{Config.MARKET_BRIEF_MINUTE:02d} {Config.TIMEZONE}")
    
    if st.button("Generate Market Brief Now"):
        try:
            response = requests.post(f"{API_URL}/market-brief")
            if response.status_code == 200:
                st.success("Market brief generation started. It will be available shortly.")
            else:
                st.error(f"Failed to start market brief generation: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main query interface
st.header("Ask a Financial Question")

if voice_input:
    st.info("🎤 Click the microphone button and speak your question")
    # Placeholder for voice input
    if st.button("🎤 Record"):
        st.info("Recording... (functionality to be implemented)")
        # TODO: Implement actual voice recording and STT
        # For now, we'll just use a text input as fallback
        query = st.text_input("Question (fallback)", placeholder="What's our risk exposure in Asia tech stocks today?")
else:
    query = st.text_input("Question", placeholder="What's our risk exposure in Asia tech stocks today?")

# Process the query
if st.button("Ask") or query:
    if query:
        st.info("Processing your question...")
        
        try:
            # Get output preference from session state
            output_preference = st.session_state.get("output_preference", "Text Only")
            
            # Determine if voice output is needed based on preference
            need_voice_output = output_preference in ["Voice (English)", "Both Text and Voice"]
            
            # Create the request payload
            payload = {
                "query": query,
                "voice_input": voice_input,
                "voice_output": need_voice_output,
                "language": "auto"  # For automatic language detection (Hindi/English)
            }
            
            # Send the request to the API
            response = requests.post(f"{API_URL}/query", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display the response based on preference
                st.subheader("Answer")
                
                # Always show text for 'Text Only' or 'Both' options
                if output_preference in ["Text Only", "Both Text and Voice"]:
                    st.write(result["response"])
                
                # Show audio for 'Voice' or 'Both' options
                if need_voice_output and result.get("audio_url"):
                    # Add a message when only voice is requested
                    if output_preference == "Voice (English)":
                        st.info("Playing voice response")
                    st.audio(result["audio_url"], format="audio/mp3")
                
                # Display sources if available
                if result.get("sources") and len(result["sources"]) > 0:
                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
                
                # Display processing time
                st.caption(f"Processing time: {result['processing_time']:.2f} seconds")
            
            else:
                st.error(f"Error processing query: {response.text}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question")

# Footer
st.markdown("---")
st.caption(f"© {datetime.now().year} Finance Agent | Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run the app
if __name__ == "__main__":
    logger.info("Streamlit app started")
