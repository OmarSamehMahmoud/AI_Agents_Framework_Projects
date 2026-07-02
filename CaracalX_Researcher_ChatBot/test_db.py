import psycopg

ports_to_try = [5442, 5432]

for port in ports_to_try:
    try:
        print(f"Attempting to connect to localhost:{port} (timeout 5s)...")
        conn = psycopg.connect(
            host="localhost",
            port=port,
            user="postgres",
            password="Omar1996",
            dbname="postgres",
            connect_timeout=5  # هيقفل الاتصال لو مفيش رد خلال 5 ثواني
        )
        print(f"✅ SUCCESS: Connected to PostgreSQL on port {port}!")
        conn.close()
        break
    except Exception as e:
        print(f"❌ Failed on port {port}. Error: {e}\n")