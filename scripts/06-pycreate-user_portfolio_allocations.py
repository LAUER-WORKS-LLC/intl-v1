import psycopg2
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# Connect to database
conn = psycopg2.connect(
    dbname=os.getenv("RDS_DB"),
    user=os.getenv("RDS_USER"),
    password=os.getenv("RDS_PASSWORD"),
    host=os.getenv("RDS_HOST"),
    port="5432"
)

print("Connected to RDS")

try:
    # Create cursor and execute schema
    cur = conn.cursor()
    print("Creating user_portfolio_allocations table...")
    cur.execute(open("../sql/06-create-user_portfolio_allocations.sql", "r").read())
    conn.commit()
    print("User_portfolio_allocations table created successfully!")
    
except Exception as e:
    print(f"Error creating user_portfolio_allocations table: {e}")
    conn.rollback()
    
finally:
    # Clean up
    cur.close()
    conn.close()
    print("Connection closed")
