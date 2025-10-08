import psycopg2
from dotenv import load_dotenv
import os
import sys

def test_connection():
    """Test connection to AWS RDS database and provide detailed error information."""
    
    # Load environment variables
    try:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))
        print("✓ Environment variables loaded successfully")
    except Exception as e:
        print(f"✗ Error loading .env file: {e}")
        return False
    
    # Check if required environment variables exist
    required_vars = ["RDS_DB", "RDS_USER", "RDS_PASSWORD", "RDS_HOST"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file contains all required variables:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return False
    
    print("✓ All required environment variables found")
    
    # Display connection details (without password)
    print(f"Database: {os.getenv('RDS_DB')}")
    print(f"User: {os.getenv('RDS_USER')}")
    print(f"Host: {os.getenv('RDS_HOST')}")
    print(f"Port: {os.getenv('RDS_PORT', '5432')}")
    
    # Test connection
    try:
        print("\nAttempting to connect to database...")
        
        conn = psycopg2.connect(
            dbname=os.getenv("RDS_DB"),
            user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASSWORD"),
            host=os.getenv("RDS_HOST"),
            port=os.getenv("RDS_PORT", "5432"),
            connect_timeout=10  # 10 second timeout
        )
        
        print("✓ Successfully connected to database!")
        
        # Test a simple query
        try:
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"✓ Database version: {version[0]}")
            cur.close()
        except Exception as e:
            print(f"⚠ Warning: Connected but query failed: {e}")
        
        conn.close()
        print("✓ Connection closed successfully")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"✗ Connection failed: {e}")
        
        # Provide specific error guidance
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            print("  → Check if your RDS instance is running and accessible")
            print("  → Verify security group allows connections from your IP")
        elif "authentication" in error_msg or "password" in error_msg:
            print("  → Check your username and password in .env file")
        elif "database" in error_msg and "does not exist" in error_msg:
            print("  → Check your database name in .env file")
        elif "could not connect" in error_msg:
            print("  → Check your host/endpoint in .env file")
            print("  → Verify RDS instance is in 'available' state")
        elif "refused" in error_msg:
            print("  → Check if port 5432 is correct")
            print("  → Verify security group allows port 5432")
        
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing AWS RDS Database Connection")
    print("=" * 40)
    
    success = test_connection()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Connection test PASSED")
        sys.exit(0)
    else:
        print("✗ Connection test FAILED")
        sys.exit(1)
