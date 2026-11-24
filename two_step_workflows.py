import statistics as stats
from pydantic import BaseModel, Field
from gemini_helper import ask_gemini, format_prompt, print_gemini


# --- Pydantic Schemas for Structured Step 1 Outputs ---
class DatasetAnalysis(BaseModel):
    rounded_mean: int = Field(description="The nearest integer value of the arithmetic mean")
    spread_summary: str = Field(description="A concise 1-sentence summary of the number spread")
    is_skewed: bool = Field(description="True if outliers heavily skew the mean")


class QueryDiagnosis(BaseModel):
    primary_bottleneck: str = Field(description="The main performance bottleneck in the SQL query")
    target_table: str = Field(description="The primary table needing index optimization")
    recommended_index_columns: list[str] = Field(
        description="Ordered list of column names for the composite index"
    )


# --- Workflow 1: Integer Mean -> Structured Analysis -> Facts ---
def workflow_integer_mean_facts() -> None:
    raw_input = input("\nEnter integers separated by spaces or commas (e.g., 14, 28, 42, 84): ")
    numbers = [int(item) for item in raw_input.replace(",", " ").split()]

    if not numbers:
        print("No valid integers entered.")
        return

    exact_mean = stats.mean(numbers)

    # STEP 1: Get guaranteed structured JSON output
    step1_prompt = format_prompt(
        "Analyze the integer dataset {dataset} with exact arithmetic mean {exact_mean}.",
        dataset=numbers,
        exact_mean=exact_mean,
    )

    print("\n--- Step 1: Structured Statistical Analysis ---")
    step1_result: DatasetAnalysis = ask_gemini(
        prompt=step1_prompt,
        system_instruction="You are a precise data statistician.",
        temperature=0.0,
        response_schema=DatasetAnalysis,
    )
    print(f"Structured Output -> {step1_result.model_dump_json(indent=2)}")

    # STEP 2: Feed extracted fields from Step 1 into Step 2's prompt
    step2_prompt = format_prompt(
        "Dataset Summary: {summary} (Skewed: {skewed}).\n"
        "Give 3 fascinating mathematical, scientific, or historical facts about the number {target}.",
        summary=step1_result.spread_summary,
        skewed=step1_result.is_skewed,
        target=step1_result.rounded_mean,
    )

    print(f"\n--- Step 2: Facts About {step1_result.rounded_mean} ---")
    print_gemini(
        prompt=step2_prompt,
        system_instruction="You are an engaging math and science communicator.",
        temperature=0.5,
    )


# --- Workflow 2: Slow SQL -> Structured Diagnosis -> Online DDL ---
def workflow_dba_query_optimizer() -> None:
    sql_query = input(
        "\nPaste a slow SQL query (or press Enter for sample Aurora MySQL query): "
    ).strip()

    if not sql_query:
        sql_query = (
            "SELECT o.order_id, o.total_amount, c.email "
            "FROM orders o JOIN customers c ON o.customer_id = c.id "
            "WHERE o.status = 'PENDING' AND o.created_at >= '2026-01-01' "
            "ORDER BY o.created_at DESC;"
        )

    # STEP 1: Extract structured bottleneck & index column list
    step1_prompt = format_prompt(
        "Diagnose this slow AWS Aurora MySQL 8.0 query and determine the ideal composite index:\n{query}",
        query=sql_query,
    )

    print("\n--- Step 1: Structured Query Diagnosis ---")
    diagnosis: QueryDiagnosis = ask_gemini(
        prompt=step1_prompt,
        system_instruction="You are a Senior Cloud Database Reliability Engineer (DBRE).",
        temperature=0.0,
        response_schema=QueryDiagnosis,
    )
    print(f"Structured Output -> {diagnosis.model_dump_json(indent=2)}")

    # STEP 2: Feed structured table & column fields into DDL generator
    cols_formatted = ", ".join(diagnosis.recommended_index_columns)
    step2_prompt = format_prompt(
        "Bottleneck identified: {bottleneck}.\n"
        "Generate a zero-downtime MySQL 8.0 `ALTER TABLE {table} ADD INDEX ... ALGORITHM=INPLACE, LOCK=NONE` "
        "statement covering columns ({columns}), plus the exact `DROP INDEX` rollback command.",
        bottleneck=diagnosis.primary_bottleneck,
        table=diagnosis.target_table,
        columns=cols_formatted,
    )

    print("\n--- Step 2: Production Online DDL & Rollback ---")
    print_gemini(
        prompt=step2_prompt,
        system_instruction="Output clean, production-safe SQL with brief comments.",
        temperature=0.0,
    )


if __name__ == "__main__":
    print("Select a 2-Step Structured Gemini Workflow:")
    print("1. Integer Mean -> Structured JSON -> Number Facts")
    print("2. Slow SQL Query -> Structured Diagnosis -> Online DDL")

    choice = input("\nEnter 1 or 2: ").strip()
    if choice == "1":
        workflow_integer_mean_facts()
    elif choice == "2":
        workflow_dba_query_optimizer()
    else:
        print("Invalid selection.")

