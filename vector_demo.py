from gemini_helper import ask_gemini
from vector_database_helper_gemini import (
    get_pg_connection,
    get_vector_object,
    init_vector_table,
    search_vector_objects,
    store_vector_object,
)

TABLE_NAME = "dba_runbooks"

SAMPLE_RUNBOOKS = [
    (
        "Aurora MySQL High CPU & InnoDB Lock Wait Contention",
        "Investigate InnoDB row lock waits using performance_schema.data_lock_waits and SHOW ENGINE INNODB STATUS. "
        "Often caused by unindexed UPDATE queries or long-running batch jobs blocking OLTP transactions.",
        "aurora-mysql",
        {"engine": "Aurora MySQL 8.0", "severity": 1, "jira_ref": "DBRE-104"},
    ),
    (
        "Aurora Replica Lag Spike During Heavy Read Traffic",
        "Check Aurora ReplicaLag in CloudWatch alongside BufferPoolHitRatio. Triggered by large analytical "
        "table scans on reader nodes or heavy DDL replication replay.",
        "aurora-mysql",
        {"engine": "Aurora MySQL 8.0", "severity": 2, "jira_ref": "DBRE-118"},
    ),
    (
        "PostgreSQL Connection Pool Exhaustion (FATAL: remaining connection slots)",
        "Resolve 'FATAL: remaining connection slots are reserved' by inspecting pg_stat_activity for idle-in-transaction "
        "sessions and routing serverless apps through PgBouncer or AWS RDS Proxy.",
        "postgres",
        {"engine": "PostgreSQL 16/17", "severity": 1, "jira_ref": "DBRE-142"},
    ),
    (
        "pgvector HNSW Index Build Out-of-Memory Error",
        "When CREATE INDEX USING hnsw fails or spills to disk temp files, increase maintenance_work_mem to at least 1GB "
        "for the session before building high-dimensional vector indexes.",
        "postgres",
        {"engine": "PostgreSQL 17", "severity": 3, "jira_ref": "DBRE-159"},
    ),
    (
        "IAM Database Authentication Token Expiration Failure",
        "Verify AWS Secrets Manager rotation permissions and ensure connection pools refresh SSL/TLS IAM auth tokens "
        "before the 15-minute expiration window.",
        "security",
        {"engine": "Multi-Engine", "severity": 2, "jira_ref": "DBRE-171"},
    ),
]


def seed_database() -> None:
    """Resets the demo table, creates HNSW + GIN indexes, and stores sample runbooks."""
    print("1. Initializing PostgreSQL pgvector table with HNSW + GIN indexes...")
    with get_pg_connection(register_vec=False) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
        conn.commit()

    init_vector_table(table_name=TABLE_NAME, dimensions=768)

    print("2. Vectorizing and storing DBA runbooks...")
    inserted_ids = []
    for title, content, category, meta in SAMPLE_RUNBOOKS:
        obj_id = store_vector_object(
            title=title,
            content=content,
            category=category,
            metadata=meta,
            table_name=TABLE_NAME,
        )
        inserted_ids.append(obj_id)
        print(f"   Stored ID={obj_id} [{meta['jira_ref']}] -> {title}")

    sample = get_vector_object(object_id=inserted_ids[0], table_name=TABLE_NAME)
    if sample:
        print(f"\n3. Verified direct lookup: '{sample['title']}' ({len(sample['embedding'].to_list())} dims)")


def interactive_error_lookup() -> None:
    """Prompts the user for a database error message and retrieves the closest solution."""
    print("\n--- Interactive DBA Error-to-Solution Search (Type 'exit' to quit) ---")
    print("Sample errors to try:")
    print("  - ERROR 1205 (HY000): Lock wait timeout exceeded; try restarting transaction")
    print("  - FATAL: remaining connection slots are reserved for non-replication superuser connections")
    print("  - CREATE INDEX hnsw running out of memory\n")

    while True:
        user_error = input("Paste database error or symptom: ").strip()
        if not user_error or user_error.lower() in ("exit", "quit", "q"):
            print("Exiting demo.")
            break

        matches = search_vector_objects(
            query_text=user_error,
            search_type="hybrid_rerank",
            top_k=2,
            table_name=TABLE_NAME,
        )

        print("\nClosest Runbook Matches:")
        for rank, item in enumerate(matches, start=1):
            print(f"  #{rank}: {item['title']} ({item['metadata']['jira_ref']}) | Score: {item['score']}")

        best = matches[0]
        summary = ask_gemini(
            prompt=(
                f"Database Error Reported: '{user_error}'\n"
                f"Matched Runbook ({best['metadata']['jira_ref']}): {best['title']} — {best['content']}\n\n"
                f"Provide a crisp 3-bullet triage and remediation plan for an on-call DBA."
            ),
            system_instruction="You are a Senior Database Reliability Engineer.",
            temperature=0.0,
        )
        print(f"\nRecommended Remediation:\n{summary}\n" + "-" * 60)


if __name__ == "__main__":
    seed_database()
    interactive_error_lookup()
