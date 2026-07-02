import json
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("expense_tracker")

DB_FILE = Path(__file__).resolve().parent / "expenses.json"


def load_expenses():
    if not DB_FILE.exists():
        return []

    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_expenses(expenses):
    with open(DB_FILE, "w") as f:
        json.dump(expenses, f, indent=4)


@mcp.tool
def add_expense(
    amount: float,
    category: str,
    description: str = ""
) -> str:
    """
    Add a new expense.
    """

    expenses = load_expenses()

    expenses.append({
        "amount": amount,
        "category": category,
        "description": description
    })

    save_expenses(expenses)

    return (
        f"Added expense ₹{amount} "
        f"under '{category}'."
    )


@mcp.tool
def list_expenses() -> list[dict]:
    """
    Return all expenses.
    """

    return load_expenses()


@mcp.tool
def total_expenses() -> float:
    """
    Calculate total expenses.
    """

    expenses = load_expenses()

    return sum(
        expense["amount"]
        for expense in expenses
    )


@mcp.tool
def category_summary() -> dict:
    """
    Show spending by category.
    """

    expenses = load_expenses()

    summary = {}

    for expense in expenses:

        category = expense["category"]

        summary[category] = (
            summary.get(category, 0)
            + expense["amount"]
        )

    return summary


@mcp.tool
def delete_expense(index: int) -> str:
    """
    Delete expense by index.
    """

    expenses = load_expenses()

    if index < 0 or index >= len(expenses):
        return "Invalid expense index."

    deleted = expenses.pop(index)

    save_expenses(expenses)

    return (
        f"Deleted ₹{deleted['amount']} "
        f"from {deleted['category']}."
    )


if __name__ == "__main__":
    mcp.run()