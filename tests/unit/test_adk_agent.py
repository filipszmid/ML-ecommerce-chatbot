"""Tests for the ADK ecommerce chat agent."""

from __future__ import annotations

from interface.chat.ecommerce_chat.agent import load_instruction, root_agent
from interface.chat.ecommerce_chat.tools import predict_product_category


def test_adk_agent_is_configured() -> None:
    """The ADK app exposes exactly the ecommerce root agent."""
    assert root_agent is not None
    assert root_agent.name == "ecommerce_category_agent"


def test_instruction_matches_conversational_contract() -> None:
    """The agent prompt describes conversational field collection."""
    instruction = load_instruction()

    assert "natural, conversational way" in instruction
    assert "Do not ask the user for ProductCategory" in instruction
    assert "predict_product_category" in instruction
    assert "encode female as 0 and male as 1" in instruction


def test_prediction_tool_name_is_stable() -> None:
    """The tool name remains stable for ADK function calling."""
    assert predict_product_category.__name__ == "predict_product_category"


def test_prediction_tool_rejects_missing_fields() -> None:
    """The tool catches missing values and returns an error for the LLM."""
    result = predict_product_category(
        Age=None,  # type: ignore
        Gender=1,
        AnnualIncome=None,  # type: ignore
        NumberOfPurchases=3,
        TimeSpentOnWebsite=15.0,
        LoyaltyProgram=0,
        DiscountsAvailed=1,
        PurchaseStatus=1,
    )

    assert result["status"] == "error"
    assert "CRITICAL" in result["message"]
    assert "missing required fields" in result["message"]
