# Snowflake-Streamlit-Admin-Tools
Direct Usable Code for Admin Management For Snowflake. Just Start a Streamlit App Within Snowflake and Copy paste in the Code


## 📊 Snowflake Access Management Tool

This Streamlit-based application helps **automate access management** by analyzing user access history in Snowflake and generating **SQL role scripts** that grant the **minimum required permissions**. It simplifies audit compliance, role minimization, and secure access provisioning based on historical usage.

---

### 🔍 What This Tool Does

- **Loads Snowflake query history** (`ACCESS_HISTORY`) to determine what **tables, schemas, and databases** a user has accessed in the past *N* days.
- **Identifies table-level READ/WRITE access**.
- **Detects schemas with recreated tables** (e.g., dropped and re-created in past 7 days).
- **Generates SQL scripts** to:
  - Create a new read-only role.
  - Grant `USAGE`/`SELECT` access only to needed objects.
  - Grant `SELECT ON FUTURE TABLES` for schemas with frequent table recreation.
  - Revoke existing roles (optional) and assign the new role to selected users.
- Provides a **tabular view** of:
  - All accessed databases, schemas, tables.
  - Timestamp of most recent access.
  - Current roles assigned to users.

---

### ✅ Benefits

- **Principle of Least Privilege**: Grant users only what they’ve accessed recently.
- **Security Hardening**: Detects dynamic schemas that require `FUTURE GRANTS`.
- **Operational Efficiency**: No need to manually review user activity or usage history.
- **Audit Support**: Exportable record of actual user interaction with data.

---

### ⚙️ Prerequisites

Before running this app, ensure the following:

#### 1. **Snowflake Setup**
- You must have access to:
  - `SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY`
  - `SNOWFLAKE.ACCOUNT_USAGE.GRANTS_TO_USERS`
  - `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY`
  - `SNOWFLAKE.ACCOUNT_USAGE.TABLES`
- Your role (e.g. `ACCOUNTADMIN`) must have visibility into **user-level query logs**.

#### 2. **Prebuilt Status Table**
This app depends on a **metadata table** to track whether a table has been recreated in the past 7 days.

It writes to:


