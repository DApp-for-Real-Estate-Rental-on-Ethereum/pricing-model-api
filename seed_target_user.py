import psycopg2
import random
from datetime import datetime, timedelta
import uuid

# DB Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'lotfi',
    'user': 'postgres',
    'password': '12345'
}

TARGET_EMAIL = "nitixaj335@roratu.com"

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def run():
    print(f"🚀 Starting seeding for {TARGET_EMAIL}...")
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # 0. Sync Sequences
    print("Syncing sequences...")
    sequences = ['addresses_id_seq', 'users_id_seq', 'bookings_id_seq']
    for seq in sequences:
        table = seq.replace('_id_seq', '')
        cur.execute(f"SELECT setval('{seq}', COALESCE((SELECT MAX(id) FROM {table}), 1))")

    # 1. Get or Create User
    cur.execute("SELECT id FROM users WHERE email = %s", (TARGET_EMAIL,))
    user = cur.fetchone()
    
    if not user:
        print("Creating user...")
        cur.execute("""
            INSERT INTO users (first_name, last_name, email, password, is_enabled, birthday, phone_number, rating, score, wallet_address)
            VALUES ('Target', 'User', %s, '$2a$10$dummyhash', true, '1990-01-01', 212600009999, 5.0, 100, '0xTARGETUSER')
            RETURNING id
        """, (TARGET_EMAIL,))
        user_id = cur.fetchone()[0]
    else:
        user_id = user[0]
        print(f"User found: ID {user_id}")

    # 2. Assign Host Role
    cur.execute("SELECT 1 FROM users_roles WHERE user_id = %s AND role = 'HOST'", (user_id,))
    if not cur.fetchone():
        print("Assigning HOST role...")
        cur.execute("INSERT INTO users_roles (user_id, role) VALUES (%s, 'HOST')", (user_id,))

    # 3. Create Properties (for Host Stats / Market Trends / Pricing)
    cities = ['Marrakech', 'Tangier', 'Casablanca', 'Rabat', 'Agadir']
    property_ids = []

    print("Creating properties...")
    for i, city in enumerate(cities, 1):
        # Create Address
        # Schema: address, city, country, postal_code
        cur.execute("""
            INSERT INTO addresses (address, city, country, postal_code)
            VALUES (%s, %s, 'Morocco', 20000)
            RETURNING id
        """, (f"{i} Prime St, {city}", city))
        address_id = cur.fetchone()[0]

        # Create Property
        prop_uuid = f"prop-target-{i}-{uuid.uuid4().hex[:8]}"
        title = f"Luxury {city} Retreat {i}"
        price = 500.0 + (i * 100.0) 
        
        # Schema: id, title, description, price, address_id, user_id (varchar)
        cur.execute("""
            INSERT INTO properties (
                id, title, description, price, daily_price, capacity, number_of_bedrooms, number_of_beds, number_of_bathrooms,
                address_id, user_id, status, type_id, negotiation_percentage
            ) VALUES (
                %s, %s, 'A beautiful place for AI testing.', %s, %s, 4, 2, 3, 2,
                %s, %s, 'APPROVED', 1, 0.0
            )
        """, (prop_uuid, title, price, price, address_id, str(user_id)))
        
        property_ids.append(prop_uuid)

        # 4. Create Historical Bookings for this Property
        for b in range(20):
            days_ago = random.randint(1, 365)
            stay_length = random.randint(2, 10)
            check_in = datetime.now() - timedelta(days=days_ago)
            check_out = check_in + timedelta(days=stay_length)
            total_price = price * stay_length
            
            cur.execute("""
                INSERT INTO bookings (
                    check_in_date, check_out_date, total_price, status,
                    created_at, property_id, user_id
                ) VALUES (
                    %s, %s, %s, 'COMPLETED',
                    %s, %s, 2 
                )
            """, (check_in, check_out, total_price, check_in, prop_uuid))
            
    print(f"Created {len(property_ids)} properties with historical data.")

    # 5. Create Tenant History for this User
    print("Creating tenant booking history...")
    
    # Get random other properties
    cur.execute("SELECT id FROM properties WHERE user_id != %s LIMIT 10", (str(user_id),))
    other_props = [r[0] for r in cur.fetchall()]
    
    if not other_props:
        other_props = property_ids

    # 5 Completed
    for _ in range(5):
        pid = random.choice(other_props)
        check_in = datetime.now() - timedelta(days=random.randint(30, 200))
        check_out = check_in + timedelta(days=3)
        cur.execute("""
            INSERT INTO bookings (check_in_date, check_out_date, total_price, status, created_at, property_id, user_id)
            VALUES (%s, %s, 1500, 'COMPLETED', %s, %s, %s)
        """, (check_in, check_out, check_in, pid, user_id))

    # 1 Cancelled
    pid = random.choice(other_props)
    check_in = datetime.now() - timedelta(days=random.randint(10, 50))
    cur.execute("""
        INSERT INTO bookings (check_in_date, check_out_date, total_price, status, created_at, property_id, user_id)
        VALUES (%s, %s, 1200, 'CANCELLED_BY_TENANT', %s, %s, %s)
    """, (check_in, check_in + timedelta(days=3), check_in, pid, user_id))
    
    # 1 Confirmed
    pid = random.choice(other_props)
    check_in = datetime.now() + timedelta(days=5)
    cur.execute("""
        INSERT INTO bookings (check_in_date, check_out_date, total_price, status, created_at, property_id, user_id)
        VALUES (%s, %s, 2000, 'CONFIRMED', NOW(), %s, %s)
    """, (check_in, check_in + timedelta(days=5), pid, user_id))

    print("Tenant history seeded.")
    print("✅ Done! User nitixaj335@roratu.com is now ready for AI features.")
    
    conn.close()

if __name__ == "__main__":
    run()
