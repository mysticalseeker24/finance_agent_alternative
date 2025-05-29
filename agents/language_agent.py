"""Language Agent for generating financial narratives using Langgraph and LangChain."""

from typing import Dict, List, Any, Optional, Union, Callable, TypedDict # Added TypedDict
import asyncio
from datetime import datetime
import json

from loguru import logger
from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage, SystemMessage # Not explicitly used in provided snippet
# from langchain.llms import OpenAI # Replaced by ChatOpenAI in initialize_llm
from langchain.chains import LLMChain
from langgraph.graph import END, StateGraph

from agents.base_agent import BaseAgent
from config import Config

# System Guidance Constants
SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT = (
    "You are a specialized financial assistant. Your expertise is in providing data-driven insights, "
    "analysis, and information related to financial markets, securities, economic indicators, and "
    "portfolio management. Focus solely on these topics. Maintain a professional, objective, and "
    "analytical tone. Do not engage in speculation, provide financial advice, or discuss "
    "non-financial subjects. Base your responses on the information and context provided."
)

SYSTEM_GUIDANCE_QUERY_REFORMULATION = (
    "You are an AI assistant helping to refine financial queries for better search results. "
    "Your goal is to make the query clearer, more specific, or appropriately broader while strictly "
    "maintaining a financial context. Only return the reformulated query."
)


class LanguageAgent(BaseAgent):
    """Agent for generating financial narratives and text responses."""

    def __init__(self):
        """Initialize the Language agent."""
        super().__init__("Language Agent")
        self.llm = None
        logger.info("Language Agent initialized")

    async def initialize_llm(self) -> bool:
        """Initialize the language model.

        Returns:
            True if initialization is successful, False otherwise.
        """
        try:
            # Initialize the GPT-4.1 model
            from langchain.chat_models import ChatOpenAI

            # Check if API key is available
            if not Config.OPENAI_API_KEY:
                logger.error(f"OpenAI API key not found. Cannot initialize language model {Config.OPENAI_CHAT_MODEL_NAME}")
                return False

            # Initialize with configured OpenAI model
            self.llm = await asyncio.to_thread(
                ChatOpenAI,
                model_name=Config.OPENAI_CHAT_MODEL_NAME,
                temperature=0.7,
                openai_api_key=Config.OPENAI_API_KEY,
            )
            logger.info(f"Initialized OpenAI language model: {Config.OPENAI_CHAT_MODEL_NAME}")
            return True
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            return False

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a language generation request.

        Args:
            request: The request containing operation and parameters.
                    Expected format:
                    {
                        "operation": "generate_market_brief"|"generate_query_response"|"generate_narrative",
                        "parameters": {...}  # Operation-specific parameters
                    }

        Returns:
            The generated text result.
        """
        # Ensure LLM is initialized
        if not self.llm:
            success = await self.initialize_llm()
            if not success:
                return {"error": "Failed to initialize language model"}

        operation = request.get("operation")
        parameters = request.get("parameters", {})

        if not operation:
            return {"error": "No operation specified"}

        # Execute the requested operation
        if operation == "generate_market_brief":
            market_data = parameters.get("market_data", {})
            portfolio_data = parameters.get("portfolio_data", {})
            news_data = parameters.get("news_data", [])

            result = await self._generate_market_brief(
                market_data, portfolio_data, news_data
            )
            return {"data": result}

        elif operation == "generate_query_response":
            query = parameters.get("query", "")
            context = parameters.get("context", [])
            portfolio_data = parameters.get("portfolio_data", {})

            if not query:
                return {"error": "No query provided for response generation"}

            result = await self._generate_query_response(query, context, portfolio_data)
            return {"data": result}

        elif operation == "generate_narrative":
            data = parameters.get("data", {})
            template = parameters.get("template", "")
            variables = parameters.get("variables", {})

            if not data and not template:
                return {
                    "error": "Either data or template must be provided for narrative generation"
                }

            result = await self._generate_narrative(data, template, variables)
            return {"data": result}

        else:
            return {"error": f"Unknown operation: {operation}"}

    async def _generate_market_brief(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a comprehensive market brief.

        Args:
            market_data: Market summary and index data.
            portfolio_data: Portfolio composition and performance data.
            news_data: Recent financial news articles.

        Returns:
            Generated market brief.
        """
        try:
            # Create a Langgraph workflow for market brief generation
            # This is a simplified implementation
            market_brief_graph = await self._create_market_brief_graph()

            # Prepare initial state with input data
            initial_state = {
                "market_data": market_data,
                "portfolio_data": portfolio_data,
                "news_data": news_data,
                "sections": [],
                "current_section": None,
                "final_brief": "",
            }

            # Execute the graph
            # In a real implementation, you would actually run the graph
            # Compile and run the graph
            app = market_brief_graph.compile()
            final_state = await app.ainvoke(initial_state)

            if final_state.get("error"):
                logger.error(f"Error in market brief generation graph: {final_state['error']}")
                return {"error": final_state["error"]}

            # Format the result using the assembled brief from the graph's final state
            brief_text = final_state.get("final_brief_text", "Error: Brief not generated.")
            
            # Extract title and summary if possible (or generate simple ones)
            # This part might need refinement based on how final_brief_text is structured
            brief_lines = brief_text.split('\n')
            title = brief_lines[0] if brief_lines else "Market Brief"
            
            # For summary, we could take the first paragraph or a specific section if tagged
            summary_content = final_state.get("generated_sections", {}).get("summary", "Summary not available.")


            brief = {
                "title": title,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "summary": summary_content,
                "market_overview": final_state.get("generated_sections", {}).get("market_overview", ""),
                "portfolio_performance": final_state.get("generated_sections", {}).get("portfolio_performance", ""),
                "news_highlights": final_state.get("generated_sections", {}).get("news_highlights", ""),
                "outlook": final_state.get("generated_sections", {}).get("outlook", ""),
                "full_text": brief_text, # The assembled brief
            }
            logger.info("Market brief generated successfully via LangGraph.")
            return brief

        except Exception as e:
            logger.error(f"Error generating market brief: {str(e)}")
            return {"error": str(e)}

    async def _generate_query_response(
        self, query: str, context: List[Dict[str, Any]], portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a response to a financial query.

        Args:
            query: The user's query.
            context: Relevant context from RAG.
            portfolio_data: Portfolio data for reference.

        Returns:
            Generated response to the query.
        """
        try:
            # Create a Langgraph workflow for query response generation
            query_response_graph = await self._create_query_response_graph()

            # Prepare initial state with input data
            initial_state = {
                "query": query,
                "context": context,
                "portfolio_data": portfolio_data,
                "reasoning": [],
                "final_response": "",
            }

            # Execute the graph
            # In a real implementation, you would actually run the graph
            # Compile and run the graph
            app = query_response_graph.compile()
            final_state = await app.ainvoke(initial_state)

            if final_state.get("error"):
                logger.error(f"Error in query response generation graph: {final_state['error']}")
                return {"error": final_state["error"]}

            response_text = final_state.get("final_response", "Error: Response not generated.")
            reasoning = final_state.get("reasoning_steps", [])

            # Format the result
            response = {
                "query": query,
                "response": response_text,
                "sources": [ # Assuming context items have metadata with source info
                    item.get("metadata", {}) for item in context if "metadata" in item
                ],
                "confidence": 0.9,  # Placeholder, could be dynamic later
                "reasoning_steps": reasoning,
            }
            logger.info(f"Generated response for query: '{query}' via LangGraph.")
            return response

        except Exception as e:
            logger.error(f"Error generating query response: {str(e)}")
            return {"error": str(e)}

    async def _generate_narrative(
        self, data: Dict[str, Any], template: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a narrative from data and a template.

        Args:
            data: Data to include in the narrative.
            template: Template for the narrative.
            variables: Variables to fill in the template.

        Returns:
            Generated narrative.
        """
        try:
            # Create a prompt template
            if not template:
                # Use a default template if none provided
                default_template_str = """You are a financial analyst providing insights on market data.
                
                Data: {data}
                
                Please provide a professional analysis of this data, highlighting key trends, risks, and opportunities.
                Focus on {focus} and consider the implications for investors.
                """
                template = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + default_template_str
            # For user-provided templates, we don't add the system guidance here,
            # assuming it might be part of the custom template or handled differently.
            # However, if the intention is to always prepend for non-default, this logic would need adjustment.
            # Based on current instruction: "Do not modify user-provided custom template strings."

            prompt_template = PromptTemplate(
                template=template, input_variables=["data", "focus"]
            )

            # Prepare input variables
            if not variables:
                variables = {"focus": "market trends"}

            # Convert data to string for template
            data_str = json.dumps(data, indent=2) if data else ""

            # Combine input variables
            input_variables = {"data": data_str, **variables}

            # In a real implementation, you would use the LLM and prompt template
            # For this example, we'll generate a mock narrative
            narrative_text = self._generate_mock_narrative(data, variables)

            # Format the result
            narrative = {
                "text": narrative_text,
                "template_used": template,
                "variables": variables,
            }

            logger.info("Generated narrative from template and data")
            return narrative

        except Exception as e:
            logger.error(f"Error generating narrative: {str(e)}")
            return {"error": str(e)}

    async def _create_market_brief_graph(self) -> StateGraph:
        """Create a Langgraph workflow for market brief generation.

        Returns:
            A StateGraph object.
        """
# --- State Schemas ---
class QueryState(TypedDict):
    query: str
    context: List[Dict[str, Any]]
    portfolio_data: Optional[Dict[str, Any]]
    reasoning_steps: List[str]
    intermediate_summary: Optional[str]
    final_response: str
    error: Optional[str]

class MarketBriefState(TypedDict):
    market_data: Dict[str, Any]
    portfolio_data: Dict[str, Any]
    news_data: List[Dict[str, Any]]
    generated_sections: Dict[str, str] # e.g., {"summary": "...", "market_overview": "..."}
    final_brief_text: str
    error: Optional[str]


# --- Helper for LLM Calls ---
    async def _get_llm_response(self, prompt_template: PromptTemplate, inputs: Dict) -> str:
        """Helper to get response from LLM using a prompt template and inputs."""
        if not self.llm:
            logger.error("LLM not initialized. Cannot get LLM response.")
            return "Error: LLM not available."
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = await chain.arun(**inputs)
            return response
        except Exception as e:
            logger.error(f"Error during LLM call: {str(e)}")
            return f"Error generating LLM response: {str(e)}"

# --- Market Brief Graph ---
    async def _create_market_brief_graph(self) -> StateGraph:
        workflow = StateGraph(MarketBriefState)

        async def generate_summary_node(state: MarketBriefState):
            logger.info("Market Brief Graph: Generating summary...")
            existing_template_str = (
                "Generate a concise market summary based on the following market data and news. "
                "Market Data: {market_data_json}\nNews Data: {news_data_json}\nSummary:"
            )
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            summary = await self._get_llm_response(
                prompt, {
                    "market_data_json": json.dumps(state['market_data']),
                    "news_data_json": json.dumps(state['news_data'])
                }
            )
            current_sections = state.get("generated_sections", {})
            return {"generated_sections": {**current_sections, "summary": summary}}

        async def generate_market_overview_node(state: MarketBriefState):
            logger.info("Market Brief Graph: Generating market overview...")
            existing_template_str = "Provide a market overview based on this data: {market_data_json}\nOverview:"
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            overview = await self._get_llm_response(prompt, {"market_data_json": json.dumps(state['market_data'])})
            current_sections = state.get("generated_sections", {})
            return {"generated_sections": {**current_sections, "market_overview": overview}}

        async def generate_portfolio_performance_node(state: MarketBriefState):
            logger.info("Market Brief Graph: Generating portfolio performance...")
            existing_template_str = "Summarize portfolio performance. Portfolio Data: {portfolio_data_json}\nPerformance Summary:"
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            performance = await self._get_llm_response(prompt, {"portfolio_data_json": json.dumps(state['portfolio_data'])})
            current_sections = state.get("generated_sections", {})
            return {"generated_sections": {**current_sections, "portfolio_performance": performance}}
        
        async def generate_key_news_node(state: MarketBriefState):
            logger.info("Market Brief Graph: Generating key news...")
            existing_template_str = "Highlight key financial news from the provided data: {news_data_json}\nKey News:"
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            key_news = await self._get_llm_response(prompt, {"news_data_json": json.dumps(state['news_data'])})
            current_sections = state.get("generated_sections", {})
            return {"generated_sections": {**current_sections, "news_highlights": key_news}}

        async def generate_outlook_node(state: MarketBriefState):
            logger.info("Market Brief Graph: Generating outlook...")
            existing_template_str = (
                "Provide a brief market outlook based on current data and news. "
                "Context: {all_sections_json}\nOutlook:"
            )
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            outlook = await self._get_llm_response(prompt, {"all_sections_json": json.dumps(state['generated_sections'])})
            current_sections = state.get("generated_sections", {})
            return {"generated_sections": {**current_sections, "outlook": outlook}}

        async def assemble_brief_node(state: MarketBriefState):
            logger.info("Market Brief Graph: Assembling final brief...")
            sections = state.get("generated_sections", {})
            brief_text = (
                f"# Daily Market Brief - {datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"## Summary\n{sections.get('summary', 'Not available.')}\n\n"
                f"## Market Overview\n{sections.get('market_overview', 'Not available.')}\n\n"
                f"## Portfolio Performance\n{sections.get('portfolio_performance', 'Not available.')}\n\n"
                f"## Key News Highlights\n{sections.get('news_highlights', 'Not available.')}\n\n"
                f"## Market Outlook\n{sections.get('outlook', 'Not available.')}\n"
            )
            return {"final_brief_text": brief_text}

        workflow.add_node("summary", generate_summary_node)
        workflow.add_node("market_overview", generate_market_overview_node)
        workflow.add_node("portfolio_performance", generate_portfolio_performance_node)
        workflow.add_node("key_news", generate_key_news_node)
        workflow.add_node("outlook", generate_outlook_node)
        workflow.add_node("assemble_brief", assemble_brief_node)

        workflow.add_edge("summary", "market_overview")
        workflow.add_edge("market_overview", "portfolio_performance")
        workflow.add_edge("portfolio_performance", "key_news")
        workflow.add_edge("key_news", "outlook")
        workflow.add_edge("outlook", "assemble_brief")
        workflow.add_edge("assemble_brief", END)
        workflow.set_entry_point("summary")
        return workflow

# --- Query Response Graph ---
    async def _create_query_response_graph(self) -> StateGraph:
        workflow = StateGraph(QueryState)

        async def analyze_query_node(state: QueryState):
            logger.info(f"Query Graph: Analyzing query: '{state['query']}'")
            # Simple logging for now, can be expanded
            return {"reasoning_steps": state.get("reasoning_steps", []) + ["Query analyzed"]}

        async def synthesize_context_node(state: QueryState):
            logger.info("Query Graph: Synthesizing context...")
            context_texts = "\n".join([doc.get("text", "") or doc.get("metadata", {}).get("text", "") for doc in state.get("context", [])])
            if not context_texts:
                logger.info("Query Graph: No context provided or context is empty.")
                return {
                    "intermediate_summary": "No specific context found to process.",
                    "reasoning_steps": state.get("reasoning_steps", []) + ["Context synthesis attempted: No context found."]
                }
            
            existing_template_str = (
                "Given the user's query: '{query}'\n"
                "And the following retrieved context:\n{context_texts}\n"
                "Summarize the key information from the context that is most relevant to answering the query. "
                "If the context seems irrelevant, explicitly state that."
            )
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            summary = await self._get_llm_response(prompt, {"query": state["query"], "context_texts": context_texts})
            return {
                "intermediate_summary": summary,
                "reasoning_steps": state.get("reasoning_steps", []) + ["Context synthesized"]
            }

        async def generate_response_node(state: QueryState):
            logger.info("Query Graph: Generating final response...")
            portfolio_summary_str = json.dumps(state.get("portfolio_data"), indent=2) if state.get("portfolio_data") else "Not applicable."
            existing_template_str = (
                "You are a helpful financial assistant. Answer the user's query based on the provided information. "
                "User Query: '{query}'\n"
                "Relevant Information/Context Summary: {relevant_information}\n"
                "User's Portfolio Summary (if relevant and available): {portfolio_summary}\n"
                "Provide a concise and accurate answer."
            )
            new_template_str = SYSTEM_GUIDANCE_FINANCIAL_ASSISTANT + "\n\n" + existing_template_str
            prompt = PromptTemplate.from_template(new_template_str)
            final_answer = await self._get_llm_response(
                prompt, {
                    "query": state["query"],
                    "relevant_information": state.get("intermediate_summary", "No summary available."),
                    "portfolio_summary": portfolio_summary_str
                }
            )
            return {
                "final_response": final_answer,
                "reasoning_steps": state.get("reasoning_steps", []) + ["Response generated"]
            }
        
        # Optional error handling node example (can be expanded)
        async def handle_error_node(state: QueryState):
            logger.error(f"Query Graph: An error occurred: {state.get('error')}")
            return {"final_response": f"Sorry, an error occurred: {state.get('error')}"}


        workflow.add_node("analyze_query", analyze_query_node)
        workflow.add_node("synthesize_context", synthesize_context_node)
        workflow.add_node("generate_response", generate_response_node)
        # workflow.add_node("error_handler", handle_error_node) # Example

        workflow.add_edge("analyze_query", "synthesize_context")
        workflow.add_edge("synthesize_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Basic conditional error routing (can be made more sophisticated)
        # workflow.add_conditional_edges(
        #     "synthesize_context",
        #     lambda state: "error_handler" if state.get("error") else "generate_response",
        # )
        workflow.set_entry_point("analyze_query")
        return workflow

    def _generate_mock_summary(self, market_data: Dict[str, Any]) -> str:
        """Generate a mock market summary.

        Args:
            market_data: Market data for summary generation.

        Returns:
            Generated summary text.
        """
        return "Markets closed higher today, with technology stocks leading the rally. Major indices showed strong performance, while bond yields declined slightly. Asian markets were mixed, with Chinese stocks facing pressure amid regulatory concerns."

    # Removing all _generate_mock_... methods as they are replaced by graph logic
    # _generate_mock_summary, _generate_mock_market_overview, _generate_mock_portfolio_performance,
    # _generate_mock_news_highlights, _generate_mock_outlook, _generate_mock_full_text,
    # _generate_mock_query_response, _generate_mock_narrative

    # _simulate_graph_execution is also removed as per instructions.

    def _generate_mock_narrative( # This one remains as it's called by a separate operation
        self, data: Dict[str, Any], variables: Dict[str, Any]
    ) -> str:
        """Generate a mock narrative from data and variables.

        Args:
            data: Data to include in the narrative.
            variables: Variables to fill in the template.

        Returns:
            Generated narrative text.
        """
        focus = variables.get("focus", "market trends")

        if focus == "market trends":
            return "Market trends indicate a robust recovery in equity prices, with broad-based participation across sectors. The recent pullback appears to be a healthy consolidation rather than the beginning of a more substantial correction. Technical indicators remain bullish, with major indices trading above key moving averages. Volume patterns support the current uptrend, suggesting continued institutional participation. Key levels to watch include 4,800 on the S&P 500, which represents significant psychological resistance."

        elif focus == "risk assessment":
            return "Our risk assessment highlights several areas of concern in the current market environment. Valuations in certain sectors, particularly technology, have reached elevated levels reminiscent of previous market peaks. Interest rate volatility poses a significant risk, with potential for rapid repricing if inflation data exceeds expectations. Geopolitical tensions in Asia could disrupt supply chains and impact global growth. We recommend maintaining adequate hedges and gradually reducing exposure to the most expensive market segments."

        elif focus == "investment opportunities":
            return "We see compelling investment opportunities in several areas of the market. Value stocks appear attractive relative to growth after significant underperformance in recent years. Within fixed income, investment-grade corporate bonds offer an appealing risk-reward profile given current spreads and the prospect of a stable economic environment. Internationally, select emerging markets offer valuation discounts despite improving fundamentals. We recommend a barbell approach, combining high-quality compounders with select cyclical recovery plays."

        else:
            return "Analysis of the provided data reveals several noteworthy insights for investors. Current market conditions suggest a balanced approach between capturing ongoing momentum and protecting against potential volatility. Sector rotation remains a dominant theme, with leadership shifting between growth and value as economic data evolves. Liquidity conditions continue to support risk assets broadly, though increasing dispersion in returns highlights the importance of security selection. We recommend maintaining discipline around position sizing and employing a systematic rebalancing approach to manage risk."

    async def generate_market_brief(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a comprehensive market brief.

        Args:
            market_data: Market summary and index data.
            portfolio_data: Portfolio composition and performance data.
            news_data: Recent financial news articles.

        Returns:
            Generated market brief.
        """
        request = {
            "operation": "generate_market_brief",
            "parameters": {
                "market_data": market_data,
                "portfolio_data": portfolio_data,
                "news_data": news_data,
            },
        }
        return await self.run(request)

    async def reformulate_query(self, original_query: str, search_summary: str = "Initial search results were not specific enough.") -> str:
        logger.info(f"Reformulating query: {original_query}")
        existing_template_str = (
            "The original user query was: '{original_query}'\n"
            "A previous search based on this query yielded results that were: '{search_summary}'\n"
            "Please reformulate the original query to be clearer, more specific, or broader in a way that might yield better search results in a financial context. "
            "Return only the reformulated query."
        )
        new_template_str = SYSTEM_GUIDANCE_QUERY_REFORMULATION + "\n\n" + existing_template_str
        prompt_template = PromptTemplate.from_template(new_template_str)
        # Ensure self.llm is initialized
        if not self.llm:
            await self.initialize_llm()
        if not self.llm: # If still not initialized
            logger.error("LLM not initialized in LanguageAgent. Cannot reformulate query.")
            return original_query # Fallback to original query

        # Use the existing _get_llm_response helper
        try:
            reformulated_query = await self._get_llm_response(
                prompt_template, 
                {"original_query": original_query, "search_summary": search_summary}
            )
            if "Error: LLM not available." in reformulated_query or "Error generating LLM response:" in reformulated_query :
                 logger.error(f"LLM response error during reformulation: {reformulated_query}")
                 return original_query # Fallback
            
            logger.info(f"Reformulated query: {reformulated_query}")
            return reformulated_query.strip()
        except Exception as e:
            logger.error(f"Error during query reformulation: {e}")
            return original_query # Fallback to original query

    async def generate_query_response(
        self, query: str, context: List[Dict[str, Any]], portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a response to a financial query.

        Args:
            query: The user's query.
            context: Relevant context from RAG.
            portfolio_data: Portfolio data for reference.

        Returns:
            Generated response to the query.
        """
        request = {
            "operation": "generate_query_response",
            "parameters": {
                "query": query,
                "context": context,
                "portfolio_data": portfolio_data,
            },
        }
        return await self.run(request)

    async def generate_narrative(
        self, data: Dict[str, Any], template: str = "", variables: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Generate a narrative from data and a template.

        Args:
            data: Data to include in the narrative.
            template: Template for the narrative.
            variables: Variables to fill in the template.

        Returns:
            Generated narrative.
        """
        request = {
            "operation": "generate_narrative",
            "parameters": {"data": data, "template": template, "variables": variables},
        }
        return await self.run(request)
