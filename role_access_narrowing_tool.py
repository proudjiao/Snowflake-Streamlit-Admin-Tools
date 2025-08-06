import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import re
from datetime import datetime, timezone
from collections import Counter

# --- App Configuration ---
st.set_page_config(
    page_title="Access Management - Snowflake User Table/Schema Access History",
    layout="wide"
)

if 'history_df' not in st.session_state:
    st.session_state['history_df'] = None
    st.session_state['user_grant'] = None
    st.session_state['db_access_df'] = None
    st.session_state['schema_access_df'] = None
    st.session_state['table_access_df'] = None
    st.session_state['recreated_schemas'] = None
    st.session_state['user_list'] = None
    st.session_state['days_to_look_back'] = None
    st.session_state['role_name'] = None


# --- Functions ---
def get_user_grants(session, user_list):
    df_grants = session.sql("select role as role_name, grantee_name from snowflake.account_usage.grants_to_users where deleted_on is null order by 2").to_pandas()
    df_grants = df_grants[["GRANTEE_NAME", "ROLE_NAME"]].rename(columns={"GRANTEE_NAME": "user_name", "ROLE_NAME":"role_name"})
    df_grants = (
        df_grants[df_grants["user_name"].isin(user_list)]
            .drop_duplicates()
            .sort_values(["user_name", "role_name"])
    )
    return df_grants

def get_access_history(session, user_list: list, days: int) -> pd.DataFrame:
    """
    Runs a single query to get all table read/write events for the given
    users in the specified time frame.
    """
    user_list_sql = ", ".join([f"'{user}'" for user in user_list])
    query = f"""
        SELECT
            user_name,
            query_start_time,
            f.value:objectName::string AS FQN_TABLE_NAME,
            'Read' as ACCESS_TYPE
        FROM
            snowflake.account_usage.access_history,
            LATERAL FLATTEN(input => direct_objects_accessed) f
        WHERE
            user_name IN ({user_list_sql})
            AND query_start_time >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
            AND f.value:objectDomain::string = 'Table'
            AND f.value:objectName::string NOT LIKE 'SNOWFLAKE.ACCOUNT_USAGE%'

        UNION ALL

        SELECT
            user_name,
            query_start_time,
            f.value:objectName::string AS FQN_TABLE_NAME,
            'Write' as ACCESS_TYPE
        FROM
            snowflake.account_usage.access_history,
            LATERAL FLATTEN(input => objects_modified) f
        WHERE
            user_name IN ({user_list_sql})
            AND query_start_time >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
            AND f.value:objectDomain::string = 'Table'
            AND f.value:objectName::string NOT LIKE 'SNOWFLAKE.ACCOUNT_USAGE%';
    """
    try:
        return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Could not retrieve table access history. Error: {e}")
        return pd.DataFrame()

def normalize_fqn(tablename: str, db, schema) -> str:
    # Remove quotes and uppercase
    cleaned = tablename.replace('"', '').replace('`', '').upper()
    parts = cleaned.split('.')

    if len(parts) == 3:
        return cleaned  # Already DB.SCHEMA.TABLE
    elif len(parts) == 2:
        return f"{db}.{parts[0]}.{parts[1]}"
    elif len(parts) == 1:
        return f"{db}.{schema}.{parts[0]}"
    else:
        # Too many parts? Just return cleaned to avoid crashing
        return cleaned

def refresh_table_create_drop_status(session, days: int = 7):
    """
    Refreshes the status table `table_create_drop_status_in_7_days`
    that contains all current tables and whether they are likely being recreated
    on a schedule based on recent CREATE/REPLACE or DROP events.
    """
    # -------------------------------
    # Step 0: Check if refresh is needed
    # -------------------------------
    status_table = "db.datagov_schema.TABLE_CREATE_DROP_STATUS_IN_7_DAYS"
    try:
        ts_check_query = f"""
            SELECT MAX(LAST_UPDATED) AS LAST_UPDATED
            FROM {status_table}
        """
        # st.info("begin query")
        last_updated = session.sql(ts_check_query).collect()[0]["LAST_UPDATED"]
        # st.info(last_updated)

        if last_updated:
            now = datetime.now(timezone.utc)
            # st.info(now)
            age_in_hours = (now - last_updated.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            # st.info(age_in_hours)
            if age_in_hours < 24:
                st.info(f"✅ Status table already refreshed {age_in_hours:.1f} hours ago. Skipping update.")
                return
            else:
                st.info(f"ℹ️ Status table is {age_in_hours:.1f} hours old. Refreshing...")
        else:
            st.info("ℹ️ Status table exists but no last_updated timestamp found. Refreshing...")

    except Exception as e:
        st.info(f"ℹ️ Cached Table not found — will create/refresh table that stores information about status. (~5 mins). \nError: {e}")

    
    with st.spinner(f"🔍 Querying Snowflake for CREATE/DROP TABLE statements from the past {days} days..."):
        # Step 1: Pull relevant query history
        query = f"""
            SELECT
                *
            FROM
                snowflake.account_usage.query_history
            WHERE
                start_time >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())
                AND (
                    query_text ILIKE 'CREATE OR REPLACE TABLE%' OR
                    query_text ILIKE 'DROP TABLE%' OR
                    query_text ILIKE 'CREATE TABLE%'
                )
        """
        query_df = session.sql(query).to_pandas()
    st.info(f"Retrieved Query history for {days} days")
    
    # Step 2: Extract (event_type, fqn_table, start_time)
    records = []
    pattern = re.compile(
        r"""
        \b
        (?P<action>
            CREATE\s+(OR\s+REPLACE\s+)?TABLE |     # CREATE or CREATE OR REPLACE
            DROP\s+TABLE                           # DROP
        )
        \s+
        (IF\s+(NOT\s+)?EXISTS\s+)?                # Optional IF [NOT] EXISTS (handles both CREATE and DROP)
        (?P<tablename>[a-zA-Z0-9_\.\"`]+)          # The table name
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    
    for _, row in query_df.iterrows():
        query_text = row["QUERY_TEXT"]
        start_time = row["START_TIME"]
        database_name = row["DATABASE_NAME"]
        schema_name = row["SCHEMA_NAME"]
    
        for match in pattern.finditer(query_text):
            try:
                action = match.group("action").strip().upper()
                raw_tablename = match.group("tablename")
                fqn = normalize_fqn(raw_tablename, database_name, schema_name)
                records.append((fqn, action, start_time))
            except IndexError as e:
                print(f"Regex group error in query: {query_text}\nError: {e}")
    
    # Convert to DataFrame
    events_df = pd.DataFrame(records, columns=["FQN_TABLE_NAME", "ACTION_TYPE", "START_TIME"])
    st.info(f"Extracted {len(events_df)} create/drop table events. Determining which tables are recreated...")

    # Step 3: Determine which tables are recreated
    recreated_flags = {}

    for fqn, group in events_df.groupby("FQN_TABLE_NAME"):
        actions = group.sort_values("START_TIME")["ACTION_TYPE"].tolist()

        create_like_count = sum(a in ["CREATE TABLE", "CREATE OR REPLACE TABLE"] for a in actions)
        dropped = "DROP TABLE" in actions

        is_recreated = False

        # Case 1: Created more than once → recreated
        if create_like_count >= 2:
            is_recreated = True
        # Case 2: Dropped and then created → recreated
        elif dropped and any(a in ["CREATE TABLE", "CREATE OR REPLACE TABLE"] for a in actions[actions.index("DROP TABLE") + 1:]):
            is_recreated = True

        recreated_flags[fqn] = is_recreated
    st.info(f"Classification Result: {Counter(recreated_flags.values())}. Getting a directory of all table and their recreation status...")

    # Step 4: Get all current tables
    with st.spinner("📂 Fetching all current base tables from account metadata..."):
        all_tables_df = session.sql("""
            SELECT
                table_catalog || '.' || table_schema || '.' || table_name AS fqn_table_name
            FROM
                snowflake.account_usage.tables
            where deleted is null and table_type = 'BASE TABLE'
        """).to_pandas()
        st.info(f"We have total {len(all_tables_df)} tables")

    all_tables_df['FQN_TABLE_NAME'] = all_tables_df['FQN_TABLE_NAME'].str.upper()
    all_tables_df['RECREATED_IN_PAST_7_DAYS'] = all_tables_df['FQN_TABLE_NAME'].map(recreated_flags).fillna(False)
    all_tables_df['LAST_UPDATED'] = datetime.now(timezone.utc)

    # Step 5: Save status table
    session.write_pandas(
        all_tables_df,
        auto_create_table=True,
        table_name="TABLE_CREATE_DROP_STATUS_IN_7_DAYS",
        database="db",
        schema="datagov_schema",
        overwrite=True
    )
    st.info(f"Marked All table successfully. result {Counter(all_tables_df['RECREATED_IN_PAST_7_DAYS'])} table_create_drop_status_in_7_days written to db.datagov_schema")

def check_for_recreated_tables(session, table_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Refers to the precomputed status table to determine which tables have been
    recreated (created or dropped) in the past 7 days.
    """
    if table_df.empty or 'FQN_TABLE_NAME' not in table_df.columns:
        st.info("No table data provided.")
        return [], []

    status_table = "db.datagov_schema.table_create_drop_status_in_7_days"
    st.info("Determining which tables/schema needs grants to future access...")

    try:
        status_df = session.table(status_table).to_pandas()
    except Exception as e:
        st.error(f"Unable to access status table: {e}")
        return [], []

    fqn_set = set(table_df['FQN_TABLE_NAME'].dropna().str.upper())

    matched_df = status_df[
        status_df['FQN_TABLE_NAME'].str.upper().isin(fqn_set) &
        status_df['RECREATED_IN_PAST_7_DAYS']
    ]

    recreated_tables = sorted(matched_df['FQN_TABLE_NAME'].tolist())
    recreated_schemas = sorted({fqn.split('.')[0] + '.' + fqn.split('.')[1] for fqn in recreated_tables})

    st.info(f"Found {len(recreated_tables)} tables across {len(recreated_schemas)} schemas that needs future grant.")
    if recreated_tables:
        st.write("Recreated Tables:", recreated_tables)
    if recreated_schemas:
        st.write("Recreated Schemas:", recreated_schemas)

    return recreated_schemas, recreated_tables


def generate_role_creation_script(
    role_name: str,
    warehouse_name: str,
    db_df: pd.DataFrame,
    table_df: pd.DataFrame,
    recreated_schemas: list,
    user_list: list,
    user_grant: pd.DataFrame
) -> str:
    """Generates a SQL script to create a role with access to specific tables only.
       Grants SELECT ON FUTURE TABLES for recreated schemas."""
    
    role_name = f'"{role_name.upper()}"'
    warehouse_name = f'"{warehouse_name}"'

    script = [
        "-- Script to create a read-only role for objects accessed by the user(s)",
        "USE ROLE ACCOUNTADMIN;",
        f"CREATE ROLE IF NOT EXISTS {role_name};",
        f"GRANT USAGE ON WAREHOUSE {warehouse_name} TO ROLE {role_name};"
    ]

    # Grant usage on databases
    unique_dbs = db_df['DATABASE_NAME'].unique()
    script += [f'GRANT USAGE ON DATABASE "{db}" TO ROLE {role_name};' for db in unique_dbs]

    # Grant usage on schemas + grant SELECT on tables individually
    unique_schemas = table_df[['DATABASE_NAME', 'SCHEMA_NAME']].drop_duplicates()
    for _, row in unique_schemas.iterrows():
        db = row['DATABASE_NAME']
        schema = row['SCHEMA_NAME']
        schema_fqn = f"{db}.{schema}"
        quoted_schema = f'"{db}"."{schema}"'

        script.append(f'GRANT USAGE ON SCHEMA {quoted_schema} TO ROLE {role_name};')

        # Filter relevant tables
        table_subset = table_df[
            (table_df['DATABASE_NAME'] == db) & (table_df['SCHEMA_NAME'] == schema)
        ]

        for _, table_row in table_subset.iterrows():
            table_name = table_row['TABLE_NAME']
            full_table_name = f'"{db}"."{schema}"."{table_name}"'
            script.append(f'GRANT SELECT ON TABLE {full_table_name} TO ROLE {role_name};')

        # Grant future SELECT only if schema was recreated
        if schema_fqn.upper() in [s.upper() for s in recreated_schemas]:
            script.append(f'-- Granting future select due to recreated tables in {quoted_schema}')
            script.append(f'GRANT SELECT ON FUTURE TABLES IN SCHEMA {quoted_schema} TO ROLE {role_name};')

    script.append("\n-- Grant this role to relevant users")
    for user in user_list:
        user_clean = user.replace("'", "").upper()
        grants_to_user = user_grant[user_grant["user_name"] == user_clean]["role_name"].tolist()
        for role in grants_to_user:
            script.append(f'REVOKE ROLE "{role}" FROM USER "{user_clean}";')
        script.append(f'GRANT USAGE ON WAREHOUSE {warehouse_name} TO ROLE {role_name};')
        script.append(f'GRANT ROLE {role_name} TO USER "{user_clean}";')
        script.append(f'ALTER USER "{user_clean}" SET DEFAULT_ROLE = {role_name};')
        script.append(f'ALTER USER "{user_clean}" SET DEFAULT_WAREHOUSE = {warehouse_name};')

    return "\n".join(script)


# --- Main Application Logic ---


st.title("👤 Access Management - Table Accessed in 90 Days and Role Script Automation")

try:
    session = get_active_session()
    current_user_name = get_current_username()
    current_warehouse_name = session.get_current_warehouse().replace('"', '')

    st.sidebar.header("Filter Options")
    user_input = st.sidebar.text_input("Enter User Name(s) (comma-separated)", value=current_user_name)
    days_to_look_back = st.sidebar.number_input("Days of History to Load", min_value=1, max_value=365, value=90)

    user_list = [user.strip().upper().replace("'", "\\'") for user in user_input.split(',')] if user_input else []

    st.sidebar.divider()
    st.sidebar.header("Generate Role Script")
    new_role_name_input = st.sidebar.text_input("Enter New Role Name")
    new_warehouse_name_input = st.sidebar.text_input("Enter Warehouse for Role")
    generate_script_button = st.sidebar.button("Generate Role SQL")

    if not user_list:
        st.warning("Please enter at least one user name in the sidebar to begin.")
    else:
        # Only refresh data if user input or days changed
        should_refresh_data = (
            st.session_state['history_df'] is None or
            st.session_state['user_list'] != user_list or
            st.session_state['days_to_look_back'] != days_to_look_back or
            st.session_state['user_grant'] is None
        )

        if should_refresh_data:
            with st.spinner("Loading and processing access history..."):
                history_df = get_access_history(session, user_list, days_to_look_back)
                user_grants_df = get_user_grants(session, user_list)

                if not history_df.empty:
                    split_names = history_df['FQN_TABLE_NAME'].str.split('.', expand=True)
                    history_df['DATABASE_NAME'] = split_names[0]
                    history_df['SCHEMA_NAME'] = split_names[1]
                    history_df['TABLE_NAME'] = split_names[2]
                    history_df.dropna(subset=['DATABASE_NAME', 'SCHEMA_NAME', 'TABLE_NAME'], inplace=True)
                    history_df['QUERY_START_TIME'] = pd.to_datetime(history_df['QUERY_START_TIME'])

                    db_access_df = history_df.groupby(['USER_NAME', 'DATABASE_NAME'])['QUERY_START_TIME'].max().reset_index()
                    db_access_df.rename(columns={'QUERY_START_TIME': 'LAST_ACCESSED_ON'}, inplace=True)
                    db_access_df = db_access_df.sort_values(by='LAST_ACCESSED_ON', ascending=False)

                    schema_access_df = history_df.groupby(['USER_NAME', 'DATABASE_NAME', 'SCHEMA_NAME'])['QUERY_START_TIME'].max().reset_index()
                    schema_access_df.rename(columns={'QUERY_START_TIME': 'LAST_ACCESSED_ON'}, inplace=True)
                    schema_access_df = schema_access_df.sort_values(by='LAST_ACCESSED_ON', ascending=False)

                    table_access_df = history_df.groupby(['USER_NAME', 'DATABASE_NAME', 'SCHEMA_NAME', 'TABLE_NAME', 'ACCESS_TYPE'])['QUERY_START_TIME'].max().reset_index()
                    table_access_df.rename(columns={'QUERY_START_TIME': 'LAST_ACCESSED_ON'}, inplace=True)
                    table_access_df = table_access_df.sort_values(by='LAST_ACCESSED_ON', ascending=False)

                    # Save to session_state
                    st.session_state['history_df'] = history_df
                    st.session_state['db_access_df'] = db_access_df
                    st.session_state['schema_access_df'] = schema_access_df
                    st.session_state['table_access_df'] = table_access_df
                    st.session_state['user_list'] = user_list
                    st.session_state['days_to_look_back'] = days_to_look_back
                    st.session_state['user_grant'] = user_grants_df
                else:
                    st.warning(f"No table access history found for the specified user(s) in the last {days_to_look_back} days.")
                    st.stop()

        # Display cached access dataframes
        if st.session_state['history_df'] is not None:
            st.divider()
            st.header("Databases Accessed (Most Recent First)")
            st.dataframe(st.session_state['db_access_df'], use_container_width=True)

            st.divider()
            st.header("Schemas Accessed (Most Recent First)")
            st.dataframe(st.session_state['schema_access_df'], use_container_width=True)

            st.divider()
            st.header("Tables Accessed (Most Recent First)")
            st.dataframe(st.session_state['table_access_df'], use_container_width=True)

            st.divider()
            st.header("Users Current Roles")
            st.dataframe(st.session_state['user_grant'], use_container_width=True)

        # Handle role script generation only when button is clicked
        if generate_script_button:
            st.divider()
            if not new_role_name_input:
                st.warning("Please enter a name for the new role in the sidebar.")
            elif st.session_state['history_df'] is None:
                st.warning("Please query access history first.")
            else:
                with st.spinner("Generating Script..."):
                    refresh_table_create_drop_status(session, 7)
                    recreated_schemas, recreated_tables = check_for_recreated_tables(session, st.session_state['history_df'])
                    st.session_state['recreated_schemas'] = recreated_schemas

                st.header(f"SQL Script for Role: {new_role_name_input.upper()}")
                sql_script = generate_role_creation_script(
                    new_role_name_input,
                    new_warehouse_name_input,
                    st.session_state['db_access_df'],
                    st.session_state['table_access_df'],
                    st.session_state['recreated_schemas'],
                    st.session_state['user_list'],
                    st.session_state['user_grant']
                )
                st.code(sql_script, language='sql')

except Exception as e:
    st.error("An unexpected error occurred. Please ensure you are running this app within a Snowflake environment.")
    st.error(e)
