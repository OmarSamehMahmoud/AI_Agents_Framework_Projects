import os
import sqlite3
import re
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "company.db"
MODEL_DIR  = BASE_DIR / "qwen35_model"


# ──────────────────────────────────────────────
# 1. DATABASE SETUP
# ──────────────────────────────────────────────
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS departments (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL,
            budget  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS employees (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            role          TEXT NOT NULL,
            salary        REAL NOT NULL,
            hire_date     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            title         TEXT NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            status        TEXT NOT NULL,
            budget        REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS project_assignments (
            employee_id INTEGER REFERENCES employees(id),
            project_id  INTEGER REFERENCES projects(id),
            role        TEXT NOT NULL,
            PRIMARY KEY (employee_id, project_id)
        );
    """)

    cur.execute("SELECT COUNT(*) FROM departments")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT INTO departments (name, budget) VALUES (?,?)", [
            ("Engineering", 850000), ("Marketing", 420000),
            ("HR", 310000), ("Finance", 560000), ("IT", 490000),
        ])
        cur.executemany(
            "INSERT INTO employees (name,email,department_id,role,salary,hire_date) VALUES (?,?,?,?,?,?)", [
            ("Omar Sameh",  "omarsameh@company.com",   1, "Senior Engineer",    95000, "2019-03-15"),
        ])
        cur.executemany("INSERT INTO projects (title,department_id,status,budget) VALUES (?,?,?,?)", [
            ("Mobile App Rewrite",  1, "in_progress", 200000),
        ])
        cur.executemany("INSERT INTO project_assignments (employee_id,project_id,role) VALUES (?,?,?)", [
            (1,1,"Tech Lead"),(2,1,"Developer"),(8,1,"Developer"),
        ])
    conn.commit()
    conn.close()
    print(f"✅  Database ready  →  {DB_PATH}")


# ──────────────────────────────────────────────
# 2. LOCAL MODEL LOADER
# ──────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        raise RuntimeError(
            f"Model not found at {MODEL_DIR}.\n"
            "Run:  python download_model.py"
        )

    print(f"🔄  Loading Qwen 3.5 0.8B from  {MODEL_DIR}  (first request only)…")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    print("✅  Model loaded and ready")
    return _pipeline


# ──────────────────────────────────────────────
# 3. SQL GENERATION (Improved with examples)
# ──────────────────────────────────────────────
def get_db_schema_text():
    """Return a compact schema string with current data context."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = ", ".join(
            f"{c[1]} {c[2]}{'(PK AUTOINCREMENT - DO NOT UPDATE)' if c[5] else ''}{'(UNIQUE)' if c[1]=='email' else ''}"
            for c in cur.fetchall()
        )
        lines.append(f"  {t}({cols})")
    
    # Add current data for context
    lines.append("\nDepartment IDs: 1=Engineering, 2=Marketing, 3=HR, 4=Finance, 5=IT")
    
    lines.append("\nCurrent employees (id: name): " + 
        ", ".join(f"{r[0]}:{r[1]}" for r in cur.execute("SELECT id, name FROM employees").fetchall()))
    
    lines.append("\nCurrent projects (id: title): " + 
        ", ".join(f"{r[0]}:{r[1]}" for r in cur.execute("SELECT id, title FROM projects").fetchall()))
    
    # Add project assignments context
    cur.execute("""
        SELECT pa.employee_id, e.name as employee_name, pa.project_id, p.title as project_title, pa.role
        FROM project_assignments pa
        JOIN employees e ON pa.employee_id = e.id
        JOIN projects p ON pa.project_id = p.id
    """)
    assignments = cur.fetchall()
    if assignments:
        lines.append("\nCurrent assignments (employee_id:employee_name - project_id:project_title - role):")
        for a in assignments:
            lines.append(f"  {a[0]}:{a[1]} - {a[2]}:{a[3]} - {a[4]}")
    
    conn.close()
    return "\n".join(lines)
    
SYSTEM_PROMPT = """You are an expert SQLite assistant. Generate ONE valid SQLite statement.

CRITICAL RULES:
1. For INSERT: NEVER include 'id' column (it is AUTOINCREMENT)
2. For UPDATE/DELETE: Use LIKE '%name%' for fuzzy name matching
3. NEVER update 'id' columns (they are Primary Keys)
4. For department_id: use numeric IDs (1=Engineering, 2=Marketing, 3=HR, 4=Finance, 5=IT)
5. Output ONLY the SQL statement with NO explanation, NO markdown
6. Always end with semicolon

Schema:
{schema}

EXAMPLES:

EMPLOYEE UPDATES:
User: "Update Alice salary to 100000"
SQL: UPDATE employees SET salary = 100000 WHERE name LIKE '%Alice%';

User: "Change Bob Smith role to Senior Developer"
SQL: UPDATE employees SET role = 'Senior Developer' WHERE name LIKE '%Bob%';

User: "Update Alice name to Alice Smith"
SQL: UPDATE employees SET name = 'Alice Smith' WHERE name LIKE '%Alice%';

User: "Change Alice email to alice.smith@company.com"
SQL: UPDATE employees SET email = 'alice.smith@company.com' WHERE name LIKE '%Alice%';

User: "Move Alice to Marketing department"
SQL: UPDATE employees SET department_id = 2 WHERE name LIKE '%Alice%';

User: "Update Alice hire date to 2020-01-15"
SQL: UPDATE employees SET hire_date = '2020-01-15' WHERE name LIKE '%Alice%';

PROJECT UPDATES:
User: "Update Mobile App Rewrite title to Mobile App 2.0"
SQL: UPDATE projects SET title = 'Mobile App 2.0' WHERE title LIKE '%Mobile App%';

User: "Change ERP Migration to Marketing department"
SQL: UPDATE projects SET department_id = 2 WHERE title LIKE '%ERP%';

User: "Update Brand Refresh status to completed"
SQL: UPDATE projects SET status = 'completed' WHERE title LIKE '%Brand%';

User: "Update ERP Migration budget to 400000"
SQL: UPDATE projects SET budget = 400000 WHERE title LIKE '%ERP%';

ASSIGNMENT UPDATES:
User: "Change Alice role in Mobile App Rewrite to Senior Developer"
SQL: UPDATE project_assignments SET role = 'Senior Developer' WHERE employee_id = (SELECT id FROM employees WHERE name LIKE '%Alice%') AND project_id = (SELECT id FROM projects WHERE title LIKE '%Mobile App%');

INSERT EXAMPLES:
User: "Add employee Sarah in IT with salary 80000"
SQL: INSERT INTO employees (name, email, department_id, role, salary, hire_date) VALUES ('Sarah', 'sarah@company.com', 5, 'Developer', 80000, date('now'));

User: "Create project AI Platform in Engineering with budget 500000"
SQL: INSERT INTO projects (title, department_id, status, budget) VALUES ('AI Platform', 1, 'planning', 500000);

DELETE EXAMPLES:
User: "Delete Brand Refresh project"
SQL: DELETE FROM projects WHERE title LIKE '%Brand Refresh%';
"""

def generate_sql(question: str) -> str:
    pipe      = get_pipeline()
    tokenizer = pipe.tokenizer
    schema    = get_db_schema_text()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    result = pipe(prompt)[0]["generated_text"].strip()
    
    # Debug: print what the model generated
    print(f"\n🤖 Model generated SQL:\n{result}\n")

    # Clean up the result
    result = re.sub(r"```(?:sql)?", "", result, flags=re.IGNORECASE).strip()
    result = result.split("<|im_end|>")[0].strip()
    result = result.split("<end_of_turn>")[0].strip()
    
    # Remove any trailing text after the SQL
    if ";" in result:
        result = result[:result.index(";") + 1]
    
    # Remove any "SQL:" prefix if present
    result = re.sub(r"^SQL:\s*", "", result, flags=re.IGNORECASE)

    return result


def run_sql(sql: str):
    """Execute SQL and return results."""
    print(f"🔧 Executing SQL:\n{sql}\n")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    
    try:
        cur.execute(sql)
        
        # Check if it's a SELECT query
        if sql.strip().upper().startswith("SELECT"):
            rows = [dict(r) for r in cur.fetchall()]
            cols = [d[0] for d in cur.description] if cur.description else []
            conn.close()
            return {
                "type": "select",
                "columns": cols,
                "rows": rows,
                "affected": len(rows)
            }
        else:
            # For INSERT, UPDATE, DELETE
            affected = cur.rowcount
            conn.commit()
            conn.close()
            
            print(f"✅ SQL executed successfully. Affected rows: {affected}")
            
            return {
                "type": "modify",
                "affected": affected,
                "operation": sql.strip().split()[0].upper()
            }
    except Exception as e:
        conn.rollback()
        conn.close()
        print(f"❌ SQL Error: {e}")
        raise e


def format_answer(question: str, sql: str, result: dict) -> str:
    """Turn query results into a readable sentence."""
    pipe      = get_pipeline()
    tokenizer = pipe.tokenizer
    
    if result["type"] == "select":
        if not result["rows"]:
            return "No results found for that query."

        header = " | ".join(result["columns"])
        body   = "\n".join(" | ".join(str(r[c]) for c in result["columns"]) for r in result["rows"][:20])
        table  = f"{header}\n{body}"
        suffix = f"\n(showing {len(result['rows'])} row{'s' if len(result['rows'])!=1 else ''})"

        messages = [
            {"role": "user", "content": f"Question: {question}\n\nSQL result:\n{table}{suffix}\n\nWrite a short, clear answer."}
        ]
    else:
        operation = result["operation"]
        affected = result["affected"]
        
        # Get fresh data to verify the change
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        verification = ""
        if "UPDATE" in sql.upper() or "DELETE" in sql.upper():
            # Extract table name
            match = re.search(r'(?:UPDATE|FROM)\s+(\w+)', sql, re.IGNORECASE)
            if match:
                table = match.group(1)
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                verification = f"\n\nVerification: {table} table now has {count} records."
        
        conn.close()
        
        messages = [
            {"role": "user", "content": f"Question: {question}\n\nOperation: {operation}\nAffected rows: {affected}{verification}\n\nWrite a short confirmation."}
        ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    answer = pipe(prompt)[0]["generated_text"].strip()
    answer = answer.split("<|im_end|>")[0].strip()
    answer = answer.split("<end_of_turn>")[0].strip()
    
    if not answer:
        if result["type"] == "select":
            return f"Found {len(result['rows'])} result(s)."
        else:
            if result["affected"] == 0:
                return f"⚠️ No rows were affected. The record might not exist or the name doesn't match."
            return f"✅ Successfully {result['operation'].lower()}ed {result['affected']} row(s)."
    
    return answer


# ──────────────────────────────────────────────
# 4. SCHEMA HELPER
# ──────────────────────────────────────────────
def get_schema_dict():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t] = [{"name": c[1], "type": c[2], "pk": bool(c[5])} for c in cur.fetchall()]
    conn.close()
    return schema


def get_database_dump():
    """Get a complete dump of all tables with sample data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    
    dump = {}
    for table in tables:
        cur.execute(f"SELECT * FROM {table} LIMIT 50")
        rows = [dict(r) for r in cur.fetchall()]
        
        # Get column names in correct order
        columns = [d[0] for d in cur.description] if cur.description else []
        
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        total = cur.fetchone()[0]
        
        dump[table] = {
            "total_rows": total,
            "columns": columns,  # ← أضفنا ده
            "sample_data": rows
        }
    
    conn.close()
    return dump
# ──────────────────────────────────────────────
# 5. ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/schema")
def schema():
    return jsonify(get_schema_dict())


@app.route("/api/chat", methods=["POST"])
def chat():
    data     = request.get_json() or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "Empty message"}), 400

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        return jsonify({
            "error": "Model not downloaded yet.\nRun:  python download_model.py"
        }), 503

    try:
        sql    = generate_sql(question)
        result = run_sql(sql)
        answer = format_answer(question, sql, result)
        
        response = {
            "answer": answer,
            "sql": sql,
            "operation_type": result["type"]
        }
        
        if result["type"] == "select":
            response["columns"] = result["columns"]
            response["rows"] = result["rows"][:100]
        else:
            response["affected_rows"] = result["affected"]
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def raw_query():
    data = request.get_json() or {}
    sql  = data.get("sql", "").strip()
    if not sql:
        return jsonify({"error": "Empty SQL"}), 400
    
    try:
        result = run_sql(sql)
        
        if result["type"] == "select":
            return jsonify({
                "type": "select",
                "columns": result["columns"],
                "rows": result["rows"]
            })
        else:
            return jsonify({
                "type": "modify",
                "operation": result["operation"],
                "affected": result["affected"]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dump")
def dump():
    """Get complete database dump."""
    try:
        dump_data = get_database_dump()
        return jsonify(dump_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def status():
    model_ready = MODEL_DIR.exists() and any(MODEL_DIR.iterdir())
    return jsonify({
        "model_dir":   str(MODEL_DIR),
        "model_ready": model_ready,
        "db_path":     str(DB_PATH),
    })


if __name__ == "__main__":
    init_database()

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print("⚠️   Model not found!  Run:  python download_model.py")
    else:
        print(f"✅  Model found at  {MODEL_DIR}")

    print("🚀  Starting server  →  http://localhost:5000")
    app.run(debug=False, port=5000)