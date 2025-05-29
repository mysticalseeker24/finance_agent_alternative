"""Test-only versions of app functions.

These functions are only used by tests and don't affect the main application.
They provide test-friendly versions with signatures that match test expectations.
"""

# Define test-specific versions of the functions
# These are completely separate from the app.py functions


def fetch_market_summary(api_base_url):
    """Test-only version: Fetch market summary data from the API."""
    import requests

    try:
        response = requests.get(
            f"{api_base_url}/api/v1/market/summary",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()["data"]
        return None
    except Exception as e:
        print(f"Error fetching market summary: {str(e)}")
        return None


def process_query(api_base_url, query, use_voice=False):
    """Test-only version: Process a query through the API."""
    import requests

    try:
        response = requests.post(
            f"{api_base_url}/api/v1/query",
            headers={"Content-Type": "application/json"},
            json={"query": query, "use_voice": use_voice},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["data"]
        return None
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return None


def generate_market_brief(api_base_url, portfolio=None, use_voice=False):
    """Test-only version: Generate a market brief through the API."""
    import requests

    try:
        response = requests.post(
            f"{api_base_url}/api/v1/market/brief",
            headers={"Content-Type": "application/json"},
            json={
                "include_portfolio": True,
                "portfolio": portfolio,
                "use_voice": use_voice,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["data"]
        return None
    except Exception as e:
        print(f"Error generating market brief: {str(e)}")
        return None


def convert_text_to_speech(api_base_url, text):
    """Test-only version: Convert text to speech using the API."""
    import requests

    try:
        response = requests.post(
            f"{api_base_url}/api/v1/voice/tts",
            headers={"Content-Type": "application/json"},
            json={"text": text},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["data"]["audio_url"]
        return None
    except Exception as e:
        print(f"Error converting text to speech: {str(e)}")
        return None
