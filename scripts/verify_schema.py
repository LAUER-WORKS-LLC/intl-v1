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

try:
    cur = conn.cursor()
    
    # Get all tables in the public schema
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    tables = cur.fetchall()
    
    if not tables:
        print("No tables found in the database")
    else:
        print(f"Found {len(tables)} table(s) in the database:")
        print("=" * 60)
        
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            print("-" * 40)
            
            # Get table structure for this table
            cur.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cur.fetchall()
            
            if columns:
                print(f"{'Column':<20} {'Type':<20} {'Nullable':<10} {'Default':<15}")
                print("-" * 65)
                
                for col in columns:
                    col_name = col[0]
                    data_type = col[1]
                    is_nullable = 'YES' if col[2] == 'YES' else 'NO'
                    default = col[3] if col[3] else ''
                    max_length = col[4]
                    
                    # Format data type with length if applicable
                    if max_length and data_type in ['character varying', 'varchar', 'char']:
                        data_type = f"{data_type}({max_length})"
                    
                    print(f"{col_name:<20} {data_type:<20} {is_nullable:<10} {default:<15}")
            else:
                print("No columns found")
        
        print("\n" + "=" * 60)
        print("Database schema verification complete!")
        
except Exception as e:
    print(f"Error: {e}")
    
finally:
    cur.close()
    conn.close()
