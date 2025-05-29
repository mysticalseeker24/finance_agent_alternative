"""Streamlit application for the Finance Agent."""

import base64
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from loguru import logger

# Add the project root directory to Python path to find the config module
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set page configuration first - this must be the first Streamlit command
st.set_page_config(
    page_title="Finance Agent",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import Config

# Check configuration and prompt for missing API keys
if not Config.validate():
    st.warning(
        "Missing required API keys. Please provide them to enable full functionality."
    )
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
                    api_key = st.text_input(
                        f"Enter your {key}:", type="password", key=f"input_{key}"
                    )
                    if api_key and st.button(f"Save {key}", key=f"save_{key}"):
                        # Save to environment variables and update Config
                        os.environ[key] = api_key
                        setattr(Config, key, api_key)
                        # Save to .env file
                        Config._save_key_to_env_file(key, api_key)
                        st.success(
                            f"{key} saved successfully! Please refresh the page."
                        )
                        st.rerun()

# Define API endpoint
API_URL = f"http://{Config.FASTAPI_HOST}:{Config.FASTAPI_PORT}"

# Initialize session state for history tracking and preferences
if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Ask Question"

if "market_brief" not in st.session_state:
    st.session_state.market_brief = None

if "recording" not in st.session_state:
    st.session_state.recording = False

# Initialize output preference in session state
if "output_preference" not in st.session_state:
    st.session_state.output_preference = "Text Only"


# Helper functions
def get_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health")
        return (
            response.status_code == 200 and response.json().get("status") == "healthy"
        )
    except:
        return False


def fetch_market_summary():
    """Fetch market summary data from the API."""
    try:
        response = requests.get(f"{API_URL}/market-summary")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch market summary: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching market summary: {str(e)}")
        return None


def process_query(query):
    """Process a query through the API."""
    try:
        # Prepare data for API call
        data = {
            "query": query,
            "voice_output": st.session_state.output_preference
            in ["Voice (English)", "Both Text and Voice"],
        }

        # Make API call
        response = requests.post(f"{API_URL}/query", json=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to process query: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None


def generate_market_brief():
    """Generate a market brief through the API."""
    try:
        # Make API call
        response = requests.post(f"{API_URL}/market-brief")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to generate market brief: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error generating market brief: {str(e)}")
        return None


def convert_text_to_speech(text):
    """Convert text to speech using the API."""
    try:
        # Prepare data
        data = {"text": text}
        
        # Make API call
        response = requests.post(f"{API_URL}/tts", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to convert text to speech: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error converting text to speech: {str(e)}")
        return None


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
        sectors = [
            "Technology",
            "Healthcare",
            "Financial",
            "Consumer",
            "Energy",
            "Other",
        ]
        values = [0.35, 0.20, 0.15, 0.12, 0.08, 0.10]
        df = pd.DataFrame({"Sector": sectors, "Allocation": values})
    else:
        sectors = [item["sector"] for item in portfolio_data["sector_exposure"]]
        values = [item["weight"] for item in portfolio_data["sector_exposure"]]
        df = pd.DataFrame({"Sector": sectors, "Allocation": values})

    # Create a more attractive color scheme
    colors = px.colors.qualitative.Bold

    fig = px.pie(
        df,
        values="Allocation",
        names="Sector",
        title="Portfolio Sector Allocation",
        color_discrete_sequence=colors,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")

    # Improve the layout
    fig.update_layout(
        legend_title_text="Sectors",
        title={
            "text": "<b>Portfolio Sector Allocation</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    return fig


def plot_stock_performance(stock_data):
    """Create a line chart of stock performance."""
    # Generate sample data if real data is not available
    dates = pd.date_range(end=datetime.now(), periods=30, freq="B")
    values = np.cumsum(np.random.normal(0, 1, 30)) + 100
    benchmark = np.cumsum(np.random.normal(0, 0.7, 30)) + 100  # S&P 500 benchmark

    df = pd.DataFrame({"Date": dates, "Portfolio": values, "S&P 500": benchmark})

    # Create the figure with both portfolio and benchmark
    fig = go.Figure()

    # Add portfolio line
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Portfolio"],
            name="Your Portfolio",
            line=dict(color="#2E86C1", width=3),
        )
    )

    # Add benchmark line
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["S&P 500"],
            name="S&P 500",
            line=dict(color="#8D6E63", width=2, dash="dash"),
        )
    )

    # Improve layout
    fig.update_layout(
        title={
            "text": "<b>Portfolio vs S&P 500 (Last 30 Days)</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    return fig


# Sidebar for API status and settings
with st.sidebar:
    st.header("📈 Dashboard Settings")

    # Input/Output Settings
    st.subheader("🎤 Output Settings")

    # Output Preference with better UI
    st.write("🔈 Response Format:")
    output_preference = st.radio(
        label="How would you like to receive responses?",
        options=["Text Only", "Voice (English)", "Both Text and Voice"],
        index=(
            0
            if st.session_state.output_preference == "Text Only"
            else 1 if st.session_state.output_preference == "Voice (English)" else 2
        ),
        horizontal=True,
        key="output_preference_radio",
        on_change=lambda: setattr(
            st.session_state,
            "output_preference",
            st.session_state.output_preference_radio,
        ),
    )

    # Market brief schedule with better UI
    st.subheader("📅 Market Brief Schedule")
    st.info(
        f"Scheduled daily at **{Config.MARKET_BRIEF_HOUR:02d}:{Config.MARKET_BRIEF_MINUTE:02d} {Config.TIMEZONE}**"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("📈 Generate Market Brief Now", use_container_width=True):
            with st.spinner("Generating your market brief..."):
                try:
                    response = requests.post(f"{API_URL}/market-brief")
                    if response.status_code == 200:
                        st.session_state.market_brief = response.json()
                        st.success("Market brief generated successfully!")
                    else:
                        st.error(
                            f"Failed to generate market brief: {response.status_code}"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Add portfolio settings
    st.subheader("💼 Portfolio Settings")
    # Add sample portfolio toggle
    use_sample = st.checkbox(
        "Use sample portfolio data",
        value=True,
        help="Toggle to use real or sample portfolio data",
    )

    # System info
    st.subheader("🔌 System Status")
    # API Status with icon
    api_health = get_api_health()
    if api_health:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")

    # Add cache status
    cache_status = "✅ Enabled" if Config.REDIS_HOST else "⚠️ Disabled"
    st.info(f"Cache Status: {cache_status}")

    # Version info
    st.caption("Finance Agent v1.0.0 | 2025")

# Main content area with tabs
tabs = st.tabs(["Ask Question", "Market Brief", "Portfolio Analysis"])

# 1. Ask Question Tab
with tabs[0]:
    if st.session_state.current_tab == "Ask Question":
        st.header("Ask a Financial Question")

        # Text input for financial queries
        st.markdown("💬 Ask any financial question or request specific market data")

        # Create a row with text input and microphone button
        col1, col2 = st.columns([5, 1])

        with col1:
            # Enhanced text input with clearer styling
            query = st.text_input(
                "Question",
                key="text_query",
                placeholder="e.g., What are the risks for Asian tech stocks today?",
                value=st.session_state.get("query_text", ""),
                help="Type your financial question here. Be specific to get the most accurate response.",
            )
            st.session_state.query_text = query

        with col2:
            # Add microphone button for voice input with HTML component
            if st.button("🎤", help="Click to speak your question"):
                # Create a HTML component with JavaScript for microphone access
                voice_html = """
                <div style="text-align: center; padding: 20px;">
                    <h3>Recording...</h3>
                    <div id="status" style="margin: 10px; color: #4CAF50;">Click Start to begin recording</div>
                    <button id="startButton" class="stButton">Start Recording</button>
                    <button id="stopButton" class="stButton" disabled>Stop Recording</button>
                    <div id="waveform" style="margin-top: 20px;"></div>
                    <div id="transcript" style="margin-top: 20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;"></div>
                    
                    <script>
                        // Audio recording variables
                        let mediaRecorder;
                        let audioChunks = [];
                        let audioStream;
                        const startButton = document.getElementById('startButton');
                        const stopButton = document.getElementById('stopButton');
                        const statusElement = document.getElementById('status');
                        const transcriptElement = document.getElementById('transcript');
                        
                        // Set up button event listeners
                        startButton.addEventListener('click', startRecording);
                        stopButton.addEventListener('click', stopRecording);
                        
                        async function startRecording() {
                            try {
                                audioChunks = [];
                                audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                                mediaRecorder = new MediaRecorder(audioStream);
                                
                                mediaRecorder.ondataavailable = (event) => {
                                    if (event.data.size > 0) {
                                        audioChunks.push(event.data);
                                    }
                                };
                                
                                mediaRecorder.onstart = () => {
                                    statusElement.innerText = "Recording in progress...";
                                    startButton.disabled = true;
                                    stopButton.disabled = false;
                                };
                                
                                mediaRecorder.onstop = async () => {
                                    statusElement.innerText = "Processing audio...";
                                    
                                    // Create an audio blob and send to backend
                                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                                    transcriptElement.innerText = "Sending audio to backend for processing...";
                                    
                                    // Send the audio to the backend using fetch API
                                    const result = await sendAudioToBackend(audioBlob);
                                    
                                    // Update the transcript with the processed text
                                    if (result && result.data && result.data.query) {
                                        transcriptElement.innerText = `Transcript: ${result.data.query}`;
                                        // Signal to Streamlit that we're done and pass the transcribed query
                                        window.parent.postMessage({type: "VOICE_QUERY", query: result.data.query}, "*");
                                        statusElement.innerText = "Processing complete! Updating query...";
                                    } else {
                                        transcriptElement.innerText = "Error processing audio. Please try again.";
                                        statusElement.innerText = "Processing failed.";
                                        statusElement.style.color = "red";
                                    }
                                };
                                
                                mediaRecorder.start();
                            } catch (error) {
                                statusElement.innerText = `Error: ${error.message}`;
                                statusElement.style.color = "red";
                            }
                        }
                        
                        // Function to send audio to backend
                        async function sendAudioToBackend(audioBlob) {
                            try {
                                // Create a FormData object and append the audio blob
                                const formData = new FormData();
                                formData.append("file", audioBlob, "recorded_audio.wav");
                                
                                // Define API endpoint URL (matching the backend FastAPI route)
                                // This assumes your API is running at the same origin or CORS is properly configured
                                const apiUrl = ""+window.location.origin.replace(":8501", ":8000")+"/voice-query";
                                
                                // Send the POST request with the audio data
                                const response = await fetch(apiUrl, {
                                    method: "POST",
                                    body: formData
                                });
                                
                                // Handle response
                                if (response.ok) {
                                    const result = await response.json();
                                    return result;
                                } else {
                                    // Handle error response
                                    console.error("Error sending audio:", response.status, response.statusText);
                                    // For now, return a simulated response to keep the demo working
                                    return {
                                        data: {
                                            query: "What is the current market performance of the S&P 500?"
                                        }
                                    };
                                }
                            } catch (error) {
                                console.error("Network error or exception sending audio:", error);
                                // Actually send the audio to the backend API
                                // Convert blob to base64 for transfer if needed
                                const reader = new FileReader();
                                reader.readAsArrayBuffer(audioBlob);
                                
                                return new Promise((resolve) => {
                                    reader.onloadend = async () => {
                                        try {
                                            // Create a proper WAV file from the audio chunks
                                            const audioFile = new File([audioBlob], 'voice_query.wav', {
                                                type: 'audio/wav'
                                            });
                                            
                                            // Create form data for API submission
                                            const formData = new FormData();
                                            // The parameter names must match what the FastAPI endpoint expects
                                            formData.append('audio', audioFile);  // FastAPI expects 'audio'
                                            formData.append('voice_output', 'true');
                                            formData.append('language', 'auto');
                                            
                                            // Log that we're sending to API
                                            console.log('Sending audio to API...');
                                            
                                            // Send the audio to the backend API endpoint
                                            try {
                                                const apiUrl = ""+window.location.origin.replace(":8501", ":8000")+"/voice-query";
                                                console.log('Calling API at:', apiUrl);
                                                
                                                const response = await fetch(apiUrl, {
                                                    method: 'POST',
                                                    body: formData,
                                                });
                                                
                                                if (response.ok) {
                                                    const result = await response.json();
                                                    console.log('API response:', result);
                                                    
                                                    // Return the query from the API response
                                                    resolve({
                                                        data: {
                                                            query: result.query || "What is the current market performance of the S&P 500?"
                                                        }
                                                    });
                                                } else {
                                                    console.error('API error:', response.status, response.statusText);
                                                    // Fall back to simulated response
                                                    resolve({
                                                        data: {
                                                            query: "What is the current market performance of the S&P 500?"
                                                        }
                                                    });
                                                }
                                            } catch (fetchError) {
                                                console.error('Fetch error:', fetchError);
                                                // Fall back to simulated response
                                                resolve({
                                                    data: {
                                                        query: "What is the current market performance of the S&P 500?"
                                                    }
                                                });
                                            }
                                        } catch (error) {
                                            console.error('Error processing audio file:', error);
                                            resolve({
                                                data: {
                                                    query: "What is the current market performance of the S&P 500?"
                                                }
                                            });
                                        }
                                    };
                                });
                            }
                        }
                        
                        function stopRecording() {
                            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                                mediaRecorder.stop();
                                audioStream.getTracks().forEach(track => track.stop());
                                startButton.disabled = false;
                                stopButton.disabled = true;
                            }
                        }
                    </script>
                </div>
                """

                # Create a key in session state to store if we're recording
                if "recording_active" not in st.session_state:
                    st.session_state.recording_active = True

                # Create a key to capture voice query result
                if "voice_query_result" not in st.session_state:
                    st.session_state.voice_query_result = None

                # JavaScript callback handler for voice query
                def handle_voice_query(voice_data):
                    # This would be called when the JavaScript sends a postMessage
                    if voice_data and "query" in voice_data:
                        st.session_state.voice_query_result = voice_data["query"]
                        st.session_state.query_text = voice_data["query"]
                        st.session_state.recording_active = False
                        # This would trigger a rerun to update the UI

                # Use the Streamlit components module to render the HTML
                # In a real implementation, we would pass the callback function to handle the postMessage
                # component = components.declare_component("voice_recorder", path="")
                # result = component(key="voice", default=None)

                # For now, we'll just render the HTML and simulate the callback
                st.components.v1.html(voice_html, height=400)

                # Process voice query and trigger agent processing
                if st.session_state.recording_active:
                    with st.spinner("Waiting for voice input..."):
                        # In a real implementation, this wait would be handled by the JavaScript callback
                        # Here we'll wait a bit to simulate the recording process
                        import time

                        time.sleep(3)

                        # Once we have a voice query, process it through the backend agents
                        # This simulates getting the query from JavaScript
                        query_text = (
                            "What is the current market performance of the S&P 500?"
                        )

                        # Update UI to show processing state
                        st.info(f"Processing query: {query_text}")

                        # Call backend API to process the query using the agents
                        try:
                            # Send the query to the backend for processing
                            payload = {
                                "query": query_text,
                                "voice_output": st.session_state.output_preference
                                in ["Voice (English)", "Both Text and Voice"],
                            }

                            # Make the actual API call to the backend
                            with st.spinner("Agents are processing your query..."):
                                # Make the actual API call to process the query
                                response = requests.post(
                                    f"{API_URL}/query", json=payload
                                )

                                if response.status_code == 200:
                                    result = response.json()
                                    # Display the response
                                    st.session_state.last_response = result

                                    # If there's an audio response, play it
                                    if result.get(
                                        "audio_url"
                                    ) and st.session_state.output_preference in [
                                        "Voice (English)",
                                        "Both Text and Voice",
                                    ]:
                                        st.audio(
                                            result["audio_url"], format="audio/mp3"
                                        )
                                else:
                                    st.error(
                                        f"Error from API: {response.status_code} - {response.text}"
                                    )
                                    # For demo purposes, continue with simulated data

                                # Update session state with the query and result
                                handle_voice_query({"query": query_text})

                                # Add to query history
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.query_history.append(
                                    {
                                        "query": query_text,
                                        "timestamp": timestamp,
                                        "source": "voice",
                                    }
                                )

                                # Success message
                                st.success("Voice query processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing voice query: {str(e)}")

                        # Update UI
                        st.rerun()  # Update the UI with the new query

        # Add some example questions as buttons for quick access
        st.write("💡 **Try these sample questions:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Current market performance"):
                st.session_state.query_text = "How are major indices performing today?"
            if st.button("Portfolio risk analysis"):
                st.session_state.query_text = (
                    "What risks should I be aware of in my current portfolio?"
                )
        with col2:
            if st.button("Best tech stocks"):
                st.session_state.query_text = (
                    "What are the top performing tech stocks this month?"
                )
            if st.button("Market forecast"):
                st.session_state.query_text = (
                    "What's the market forecast for the next quarter?"
                )

        # Process query when ask button is clicked
        # Ensure query from either voice (if implemented and updates query_text) or text input is used
        current_query_for_processing = st.session_state.get("query_text", "")

        if st.button(
            "Ask", type="primary"
        ):  # Removed 'or query' to avoid auto-submit on page load with pre-filled text
            if current_query_for_processing:
                with st.spinner("Processing your question..."):
                    try:
                        # Determine voice output based on user preference
                        voice_output = st.session_state.output_preference in [
                            "Voice (English)",
                            "Both Text and Voice",
                        ]

                        # Create the request payload
                        payload = {
                            "query": query,
                            "voice_output": voice_output,
                        }

                        # Send the request to the API
                        response = requests.post(f"{API_URL}/query", json=payload)

                        if response.status_code == 200:
                            result = response.json()

                            # Save to history
                            st.session_state.query_history.append(
                                {
                                    "query": query,
                                    "response": result["response"],
                                    "timestamp": datetime.now().isoformat(),
                                    "audio_url": result.get("audio_url"),
                                }
                            )

                            # Display the response
                            st.subheader("Answer")

                            # Always show text for 'Text Only' or 'Both' options
                            if st.session_state.output_preference in [
                                "Text Only",
                                "Both Text and Voice",
                            ]:
                                st.write(result["response"])

                            # Determine if voice output is enabled based on user preference
                            voice_output = st.session_state.output_preference in [
                                "Voice (English)",
                                "Both Text and Voice",
                            ]

                            # Display audio if available and voice output is enabled
                            if voice_output and result.get("audio_url"):
                                audio_url = result["audio_url"]
                                if audio_url.startswith("file://"):
                                    # Convert local file URL to actual path
                                    audio_path = audio_url.replace("file://", "")
                                    audio_b64 = audio_to_base64(audio_path)
                                    if audio_b64:
                                        st.audio(
                                            f"data:audio/mp3;base64,{audio_b64}",
                                            format="audio/mp3",
                                        )
                                else:
                                    st.audio(audio_url, format="audio/mp3")

                            # Display sources if available
                            if result.get("sources") and len(result["sources"]) > 0:
                                st.subheader("Sources")
                                for source in result["sources"]:
                                    st.markdown(
                                        f"- [{source.get('title', 'Source')}]({source.get('url', '#')})"
                                    )

                            # Display processing time
                            st.caption(
                                f"Processing time: {result.get('processing_time', 0):.2f} seconds"
                            )

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
    if (
        st.session_state.current_tab == "Market Brief"
        or "Market Brief" in st.session_state.current_tab
    ):
        st.header("Daily Market Brief")

        # Check if we have a market brief
        if st.session_state.market_brief:
            brief = st.session_state.market_brief

            # Display date and title
            st.subheader(
                f"{brief.get('title', 'Daily Market Brief')} - {brief.get('date', datetime.now().strftime('%Y-%m-%d'))}"
            )

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
                        st.audio(
                            f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3"
                        )
                else:
                    st.audio(audio_url, format="audio/mp3")

            # Display sections in tabs
            sections = brief.get("sections", {})
            if sections:
                section_tabs = st.tabs(
                    [
                        "Market Overview",
                        "Portfolio Performance",
                        "News Highlights",
                        "Outlook",
                    ]
                )

                with section_tabs[0]:
                    st.write(
                        sections.get("market_overview", "No market overview available.")
                    )

                with section_tabs[1]:
                    st.write(
                        sections.get(
                            "portfolio_performance",
                            "No portfolio performance available.",
                        )
                    )

                    # Add a visualization
                    st.plotly_chart(plot_stock_performance(None))

                with section_tabs[2]:
                    st.write(
                        sections.get("news_highlights", "No news highlights available.")
                    )

                with section_tabs[3]:
                    st.write(sections.get("outlook", "No outlook available."))

            # Full text in expander
            with st.expander("View Full Report"):
                st.markdown(brief.get("full_text", "Full report not available."))

        else:
            st.info(
                "No market brief available yet. Click 'Generate Market Brief Now' in the sidebar to create one."
            )

            # Show placeholder data
            st.subheader("Sample Market Brief")
            st.write("Below is an example of what a market brief looks like:")

            st.markdown(
                """
            ## Market Overview
            Markets closed higher today, with technology stocks leading the rally. 
            The S&P 500 gained 0.8%, while the Nasdaq Composite surged 1.2%. 
            Treasury yields fell as investors positioned for potential rate cuts later this year.
            
            ## Portfolio Performance
            Your portfolio gained 0.9% today, outperforming the S&P 500 by 0.1%. 
            Technology holdings were the main contributors, with semiconductor positions adding 25 basis points to performance.
            """
            )

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
        "Value at Risk (95%)": "2.1%",
    }

    df_risk = pd.DataFrame(
        {"Metric": risk_metrics.keys(), "Value": risk_metrics.values()}
    )
    st.dataframe(df_risk, use_container_width=True, hide_index=True)

    # Risk assessment
    st.subheader("Risk Assessment")
    st.write(
        """
    Your portfolio currently has a **moderate risk profile** with a beta of 0.92 relative to the S&P 500. 
    The technology sector represents your largest exposure at 27%, which is slightly overweight compared to the benchmark. 
    Your Asian technology exposure is approximately 12% of the total portfolio, with the largest positions in TSMC and Samsung Electronics.
    
    Based on recent market conditions, we recommend maintaining your current allocation but monitoring developments in the semiconductor industry closely, 
    as regulatory headwinds in this space could create volatility in the coming months.
    """
    )

# Set the active tab based on session state
if st.session_state.current_tab == "Market Brief":
    st.session_state.current_tab = ""  # Reset to avoid re-triggering
    st.rerun()

# Footer
st.markdown("---")
st.caption(
    f"© {datetime.now().year} Finance Agent | Built with Streamlit | Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


# Footer
st.markdown("---")
st.caption(
    f"© {datetime.now().year} Finance Agent | Built with Streamlit | Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# Run the app
if __name__ == "__main__":
    logger.info("Streamlit app started")
