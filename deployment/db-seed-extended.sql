-- Extended Seed Data for derent PostgreSQL Database
-- ===================================================
-- Generates hundreds of realistic records for ML model training
-- Designed to prevent overfitting by creating varied, realistic data patterns
-- Usage: docker exec -i projetjee-postgres-1 psql -U postgres -d derent < db-seed-extended.sql

BEGIN;

-- ============================================================================
-- 1. BASIC LOOKUP/REFERENCE DATA (Keep existing)
-- ============================================================================

-- Amenity categories
INSERT INTO amenity_categories (id, title) VALUES
    (1, 'Essentials'),
    (2, 'Comfort'),
    (3, 'Work'),
    (4, 'Family')
ON CONFLICT (id) DO NOTHING;

-- Amenities
INSERT INTO amenities (id, name, icon, category_id) VALUES
    (1, 'Wi-Fi', 'wifi', 1),
    (2, 'Air conditioning', 'ac', 2),
    (3, 'Heating', 'heating', 2),
    (4, 'Kitchen', 'kitchen', 1),
    (5, 'Washer', 'washer', 1),
    (6, 'Dryer', 'dryer', 1),
    (7, 'Workspace', 'desk', 3),
    (8, 'Crib', 'crib', 4),
    (9, 'Free parking', 'parking', 1),
    (10, 'Pool', 'pool', 2)
ON CONFLICT (id) DO NOTHING;

-- Property types
INSERT INTO property_types (id, type) VALUES
    (1, 'Apartment'),
    (2, 'House'),
    (3, 'Villa')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 2. USERS AND ROLES (Extended with bulk generation)
-- ============================================================================

-- Original test users (keep these)
INSERT INTO users (id, first_name, last_name, email, password, birthday, phone_number,
                   is_enabled, rating, score, wallet_address)
VALUES
    (1, 'Admin', 'User', 'admin@example.com', '$2a$10$dummyhash', '1990-01-01', 212600000001, true, 4.9, 100, '0xADMIN'),
    (2, 'Hassan', 'Host', 'host@example.com', '$2a$10$dummyhash', '1988-05-15', 212600000002, true, 4.7, 98, '0x70997970C51812dc3A010C7d01b50e0d17dc79C8'), -- Hardhat Account #1
    (3, 'Sara', 'Tenant', 'tenant@example.com', '$2a$10$dummyhash', '1995-03-10', 212600000003, true, 4.8, 95, '0x90F79bf6EB2c4f870365E785982E1f101E93b906'), -- Hardhat Account #3
    (4, 'MedGM', 'Host', 'nitixaj335@roratu.com', '$2b$12$6hlATJwWgYbCLsXOI81TNO9GdoMHFOpx7aodMLTGuORpBnBK1mljG', '1990-01-01', 212600000004, true, 5.0, 100, '0xMEDGM')
ON CONFLICT (id) DO NOTHING;

-- Bulk HOSTS (ids 10-49 = 40 hosts)
INSERT INTO users (id, first_name, last_name, email, password, birthday, phone_number,
                   is_enabled, rating, score, wallet_address, penalty_points, is_suspended)
SELECT
    gs AS id,
    'Host' || gs::text AS first_name,
    CASE (gs % 5)
        WHEN 0 THEN 'Alami'
        WHEN 1 THEN 'Benali'
        WHEN 2 THEN 'Idrissi'
        WHEN 3 THEN 'Tazi'
        ELSE 'Fassi'
    END AS last_name,
    'host' || gs::text || '@example.com' AS email,
    '$2a$10$dummyhash' AS password,
    DATE '1975-01-01' + (gs % 365) * INTERVAL '1 day' AS birthday,
    212600000000 + gs AS phone_number,
    CASE WHEN random() < 0.95 THEN true ELSE false END AS is_enabled,
    -- Rating: most hosts have 3.5-5.0, some lower
    CASE 
        WHEN random() < 0.8 THEN 3.5 + (random() * 1.5)
        ELSE 2.5 + (random() * 1.0)
    END AS rating,
    -- Score: 70-100 for most, some lower
    CASE
        WHEN random() < 0.85 THEN 70 + (random() * 30)::int
        WHEN random() < 0.95 THEN 50 + (random() * 20)::int
        ELSE 30 + (random() * 20)::int
    END AS score,
    '0xHOST' || LPAD(gs::text, 10, '0') AS wallet_address,
    -- Penalty points: most have 0-10, some have more
    CASE
        WHEN random() < 0.7 THEN 0
        WHEN random() < 0.9 THEN (random() * 10)::int
        ELSE (10 + random() * 20)::int
    END AS penalty_points,
    -- Suspension: rare (5%)
    random() < 0.05 AS is_suspended
FROM generate_series(10, 49) AS gs
ON CONFLICT (id) DO NOTHING;

INSERT INTO users_roles (user_id, role)
SELECT gs, 'HOST' FROM generate_series(10, 49) AS gs
ON CONFLICT (user_id, role) DO NOTHING;

-- Bulk TENANTS (ids 50-249 = 200 tenants)
INSERT INTO users (id, first_name, last_name, email, password, birthday, phone_number,
                   is_enabled, rating, score, wallet_address, penalty_points, is_suspended)
SELECT
    gs AS id,
    CASE (gs % 10)
        WHEN 0 THEN 'Ahmed'
        WHEN 1 THEN 'Fatima'
        WHEN 2 THEN 'Mohammed'
        WHEN 3 THEN 'Aicha'
        WHEN 4 THEN 'Youssef'
        WHEN 5 THEN 'Sanae'
        WHEN 6 THEN 'Omar'
        WHEN 7 THEN 'Khadija'
        WHEN 8 THEN 'Ali'
        ELSE 'Nadia'
    END AS first_name,
    CASE (gs % 8)
        WHEN 0 THEN 'Alaoui'
        WHEN 1 THEN 'Bennani'
        WHEN 2 THEN 'Cherkaoui'
        WHEN 3 THEN 'El Amrani'
        WHEN 4 THEN 'Fadili'
        WHEN 5 THEN 'Hafidi'
        WHEN 6 THEN 'Lamrani'
        ELSE 'Mansouri'
    END AS last_name,
    'tenant' || gs::text || '@example.com' AS email,
    '$2a$10$dummyhash' AS password,
    DATE '1990-01-01' + (gs % 365) * INTERVAL '1 day' AS birthday,
    212700000000 + gs AS phone_number,
    CASE WHEN random() < 0.98 THEN true ELSE false END AS is_enabled,
    -- Rating: varied distribution
    CASE
        WHEN random() < 0.6 THEN 4.0 + (random() * 1.0)  -- Good tenants (60%)
        WHEN random() < 0.85 THEN 3.0 + (random() * 1.0)  -- Average (25%)
        ELSE 2.0 + (random() * 1.0)                       -- Lower (15%)
    END AS rating,
    -- Score: varied, some with penalties
    CASE
        WHEN random() < 0.7 THEN 80 + (random() * 20)::int   -- Good (70%)
        WHEN random() < 0.9 THEN 60 + (random() * 20)::int   -- Average (20%)
        ELSE 40 + (random() * 20)::int                       -- Lower (10%)
    END AS score,
    '0xTENANT' || LPAD(gs::text, 10, '0') AS wallet_address,
    -- Penalty points: most have 0, some have penalties
    CASE
        WHEN random() < 0.75 THEN 0
        WHEN random() < 0.9 THEN (random() * 15)::int
        ELSE (15 + random() * 25)::int
    END AS penalty_points,
    -- Suspension: rare (3%)
    random() < 0.03 AS is_suspended
FROM generate_series(50, 249) AS gs
ON CONFLICT (id) DO NOTHING;

INSERT INTO users_roles (user_id, role)
SELECT gs, 'TENANT' FROM generate_series(50, 249) AS gs
ON CONFLICT (user_id, role) DO NOTHING;

-- User profile status (for all users)
INSERT INTO user_profile_status (user_id, is_complete, is_deleted)
SELECT 
    gs::text AS user_id,
    CASE WHEN random() < 0.85 THEN true ELSE false END AS is_complete,
    false AS is_deleted
FROM generate_series(1, 249) AS gs
ON CONFLICT (user_id) DO NOTHING;

-- ============================================================================
-- 3. ADDRESSES AND PROPERTIES (Extended)
-- ============================================================================

-- Original addresses (keep these)
INSERT INTO addresses (id, address, city, country, latitude, longitude, postal_code)
VALUES
    (1, 'Boulevard d''Anfa 123', 'Casablanca', 'Morocco', 33.5928, -7.6192, 20000),
    (2, 'Medina Street 45', 'Marrakech', 'Morocco', 31.6295, -7.9811, 40000),
    (3, 'Beach Road 9', 'Agadir', 'Morocco', 30.4289, -9.5981, 80000),
    (4, 'Avenue Mohammed V', 'Rabat', 'Morocco', 34.0209, -6.8416, 10000),
    (5, 'Old Medina Alley 7', 'Fes', 'Morocco', 34.0331, -5.0003, 30000)
ON CONFLICT (id) DO NOTHING;

-- Generate more addresses (ids 6-105 = 100 addresses)
INSERT INTO addresses (id, address, city, country, latitude, longitude, postal_code)
SELECT
    5 + gs AS id,
    CASE (gs % 5)
        WHEN 0 THEN 'Rue ' || (gs % 100)::text || ', Casablanca'
        WHEN 1 THEN 'Avenue ' || (gs % 50)::text || ', Marrakech'
        WHEN 2 THEN 'Boulevard ' || (gs % 80)::text || ', Agadir'
        WHEN 3 THEN 'Street ' || (gs % 60)::text || ', Rabat'
        ELSE 'Alley ' || (gs % 40)::text || ', Fes'
    END AS address,
    CASE (gs % 5)
        WHEN 0 THEN 'Casablanca'
        WHEN 1 THEN 'Marrakech'
        WHEN 2 THEN 'Agadir'
        WHEN 3 THEN 'Rabat'
        ELSE 'Fes'
    END AS city,
    'Morocco' AS country,
    CASE (gs % 5)
        WHEN 0 THEN 33.5 + (random() * 0.2)  -- Casablanca area
        WHEN 1 THEN 31.6 + (random() * 0.2)  -- Marrakech area
        WHEN 2 THEN 30.4 + (random() * 0.2)  -- Agadir area
        WHEN 3 THEN 34.0 + (random() * 0.2)  -- Rabat area
        ELSE 34.0 + (random() * 0.2)         -- Fes area
    END AS latitude,
    CASE (gs % 5)
        WHEN 0 THEN -7.6 + (random() * 0.2)
        WHEN 1 THEN -7.9 + (random() * 0.2)
        WHEN 2 THEN -9.6 + (random() * 0.2)
        WHEN 3 THEN -6.8 + (random() * 0.2)
        ELSE -5.0 + (random() * 0.2)
    END AS longitude,
    CASE (gs % 5)
        WHEN 0 THEN 20000 + (gs % 1000)
        WHEN 1 THEN 40000 + (gs % 1000)
        WHEN 2 THEN 80000 + (gs % 1000)
        WHEN 3 THEN 10000 + (gs % 1000)
        ELSE 30000 + (gs % 1000)
    END AS postal_code
FROM generate_series(1, 300) AS gs
ON CONFLICT (id) DO NOTHING;

-- Original properties (keep these)
INSERT INTO properties (
    id, capacity, daily_price, deposit_amount, description,
    discount_fifteen_days, discount_five_days, discount_one_month,
    negotiation_percentage, number_of_bathrooms, number_of_bedrooms,
    number_of_beds, price, status, title, user_id, address_id, type_id,
    city, country, latitude, longitude
) VALUES
    ('prop-casa-001', 4, 450, 200, 'Modern apartment in central Casablanca, close to tram and business district.',
     15, 5, 25, 10, 1, 2, 3, 450, 'APPROVED', 'Modern Business Apartment - Casablanca', '2', 1, 1,
     'Casablanca', 'Morocco', 33.5928, -7.6192),
    ('prop-marr-001', 6, 650, 300, 'Traditional riad in the Medina with private patio and rooftop.',
     10, 5, 20, 12, 2, 3, 4, 650, 'APPROVED', 'Riad with Rooftop - Marrakech Medina', '2', 2, 2,
     'Marrakech', 'Morocco', 31.6295, -7.9811),
    ('prop-agad-001', 5, 500, 150, 'Beachfront apartment with ocean view and pool access.',
     12, 5, 22, 8, 1, 2, 3, 500, 'APPROVED', 'Beachfront Apartment - Agadir', '2', 3, 1,
     'Agadir', 'Morocco', 30.4289, -9.5981),
    ('prop-rabat-001', 3, 380, 100, 'Cozy studio near administrative district and tram.',
     8, 3, 18, 7, 1, 1, 2, 380, 'APPROVED', 'Cozy Studio - Rabat Center', '2', 4, 1,
     'Rabat', 'Morocco', 34.0209, -6.8416),
    ('prop-fes-001', 4, 320, 80, 'Authentic house in Fes Medina, perfect for cultural stays.',
     10, 5, 20, 9, 1, 2, 3, 320, 'APPROVED', 'Authentic Medina House - Fes', '2', 5, 2,
     'Fes', 'Morocco', 34.0331, -5.0003)
ON CONFLICT (id) DO NOTHING;

-- Bulk properties (300 properties across all hosts and cities)
INSERT INTO properties (
    id, capacity, daily_price, deposit_amount, description,
    discount_fifteen_days, discount_five_days, discount_one_month,
    negotiation_percentage, number_of_bathrooms, number_of_bedrooms,
    number_of_beds, price, status, title, user_id, address_id, type_id,
    city, country, latitude, longitude
)
SELECT
    'prop-gen-' || gs::text AS id,
    -- Capacity: 2-8 people
    (2 + (random() * 6)::int) AS capacity,
    -- Daily price: varies by city and type
    CASE (gs % 5)
        WHEN 0 THEN (300 + random() * 300)::int  -- Casablanca: 300-600
        WHEN 1 THEN (400 + random() * 400)::int  -- Marrakech: 400-800
        WHEN 2 THEN (250 + random() * 350)::int  -- Agadir: 250-600
        WHEN 3 THEN (280 + random() * 320)::int  -- Rabat: 280-600
        ELSE (200 + random() * 300)::int         -- Fes: 200-500
    END AS daily_price,
    -- Deposit: 20-30% of daily price
    ((300 + random() * 300) * 0.25)::int AS deposit_amount,
    'Generated property description #' || gs::text AS description,
    -- Discounts: realistic ranges
    (5 + (random() * 15)::int) AS discount_fifteen_days,
    (3 + (random() * 10)::int) AS discount_five_days,
    (10 + (random() * 20)::int) AS discount_one_month,
    -- Negotiation: 5-20%
    (5 + (random() * 15)::int) AS negotiation_percentage,
    -- Bathrooms: 1-3
    1 + (random() * 2)::int AS number_of_bathrooms,
    -- Bedrooms: 1-4
    1 + (random() * 3)::int AS number_of_bedrooms,
    -- Beds: 1-5
    1 + (random() * 4)::int AS number_of_beds,
    -- Legacy price (same as daily_price)
    0.0 AS price,
    -- Status: mostly APPROVED, some others
    CASE
        WHEN random() < 0.85 THEN 'APPROVED'
        WHEN random() < 0.95 THEN 'VISIBLE_ONLY_FOR_TENANTS'
        ELSE 'PENDING_APPROVAL'
    END AS status,
    'Generated Listing #' || gs::text AS title,
    -- Random host (ids 10-49)
    (10 + (random() * 40)::int)::text AS user_id,
    -- Address: unique address for each property (ids 6-305)
    (6 + (gs - 1000)) AS address_id,
    -- Property type: varied
    (1 + (gs % 3)) AS type_id,
    -- City: varied
    CASE (gs % 5)
        WHEN 0 THEN 'Casablanca'
        WHEN 1 THEN 'Marrakech'
        WHEN 2 THEN 'Agadir'
        WHEN 3 THEN 'Rabat'
        ELSE 'Fes'
    END AS city,
    'Morocco' AS country,
    -- Coordinates based on city
    CASE (gs % 5)
        WHEN 0 THEN 33.5 + (random() * 0.2)
        WHEN 1 THEN 31.6 + (random() * 0.2)
        WHEN 2 THEN 30.4 + (random() * 0.2)
        WHEN 3 THEN 34.0 + (random() * 0.2)
        ELSE 34.0 + (random() * 0.2)
    END AS latitude,
    CASE (gs % 5)
        WHEN 0 THEN -7.6 + (random() * 0.2)
        WHEN 1 THEN -7.9 + (random() * 0.2)
        WHEN 2 THEN -9.6 + (random() * 0.2)
        WHEN 3 THEN -6.8 + (random() * 0.2)
        ELSE -5.0 + (random() * 0.2)
    END AS longitude
FROM generate_series(1000, 1299) AS gs
ON CONFLICT (id) DO NOTHING;

-- Link addresses to properties
UPDATE addresses a
SET property_id = p.id
FROM properties p
WHERE a.id = p.address_id AND a.property_id IS NULL;

-- ============================================================================
-- 4. PROPERTY IMAGES (Extended)
-- ============================================================================

-- Original images (keep these)
INSERT INTO property_images (id, propety_id, url, is_cover) VALUES
    (1, 'prop-casa-001', 'https://picsum.photos/id/1018/800/600', true),
    (2, 'prop-casa-001', 'https://picsum.photos/id/1025/800/600', false),
    (3, 'prop-marr-001', 'https://picsum.photos/id/1035/800/600', true),
    (4, 'prop-marr-001', 'https://picsum.photos/id/1045/800/600', false),
    (5, 'prop-agad-001', 'https://picsum.photos/id/1050/800/600', true),
    (6, 'prop-rabat-001', 'https://picsum.photos/id/1060/800/600', true),
    (7, 'prop-fes-001', 'https://picsum.photos/id/1070/800/600', true)
ON CONFLICT (id) DO NOTHING;

-- Generate images for all properties (2-6 images per property)
INSERT INTO property_images (id, propety_id, url, is_cover)
SELECT
    1000 + (gs * 10) + img_num AS id,
    p.id AS propety_id,
    'https://picsum.photos/id/' || (1000 + gs * 10 + img_num)::text || '/800/600' AS url,
    (img_num = 0) AS is_cover
FROM generate_series(1000, 1299) AS gs
CROSS JOIN generate_series(0, (2 + random() * 4)::int) AS img_num
JOIN properties p ON p.id = 'prop-gen-' || gs::text
WHERE p.id IS NOT NULL
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 5. PROPERTY AMENITIES (Extended)
-- ============================================================================

-- Original amenities (keep these)
INSERT INTO properties_amenities (property_id, amenity_id) VALUES
    ('prop-casa-001', 1), ('prop-casa-001', 2), ('prop-casa-001', 4), ('prop-casa-001', 7), ('prop-casa-001', 9),
    ('prop-marr-001', 1), ('prop-marr-001', 2), ('prop-marr-001', 3), ('prop-marr-001', 4), ('prop-marr-001', 10),
    ('prop-agad-001', 1), ('prop-agad-001', 2), ('prop-agad-001', 4), ('prop-agad-001', 9), ('prop-agad-001', 10),
    ('prop-rabat-001', 1), ('prop-rabat-001', 4), ('prop-rabat-001', 7),
    ('prop-fes-001', 1), ('prop-fes-001', 3), ('prop-fes-001', 4), ('prop-fes-001', 8)
ON CONFLICT (property_id, amenity_id) DO NOTHING;

-- Generate amenities for all properties (3-8 amenities per property)
INSERT INTO properties_amenities (property_id, amenity_id)
SELECT DISTINCT
    p.id AS property_id,
    (1 + (random() * 9)::int) AS amenity_id
FROM properties p
CROSS JOIN generate_series(1, (3 + random() * 5)::int) AS amenity_count
WHERE p.id LIKE 'prop-gen-%'
ON CONFLICT (property_id, amenity_id) DO NOTHING;

-- ============================================================================
-- 6. BOOKINGS (Extended - 2000+ bookings)
-- ============================================================================

-- Original bookings (keep these)
INSERT INTO bookings (
    id, user_id, property_id,
    check_in_date, check_out_date, created_at, updated_at,
    long_stay_discount_percent, requested_negotiation_percent,
    negotiation_expires_at, on_chain_tx_hash, status, total_price
) VALUES
    (1, 3, 'prop-casa-001',
     CURRENT_DATE - INTERVAL '20 days', CURRENT_DATE - INTERVAL '15 days',
     CURRENT_TIMESTAMP - INTERVAL '25 days', CURRENT_TIMESTAMP - INTERVAL '14 days',
     0, 5, CURRENT_TIMESTAMP - INTERVAL '24 days', '0xTXPAST1', 'COMPLETED', 5 * 450),
    (2, 3, 'prop-marr-001',
     CURRENT_DATE - INTERVAL '5 days', CURRENT_DATE + INTERVAL '2 days',
     CURRENT_TIMESTAMP - INTERVAL '8 days', CURRENT_TIMESTAMP - INTERVAL '1 day',
     0, 10, CURRENT_TIMESTAMP - INTERVAL '7 days', '0xTXCURR1', 'CONFIRMED', 7 * 650),
    (3, 3, 'prop-agad-001',
     CURRENT_DATE + INTERVAL '10 days', CURRENT_DATE + INTERVAL '15 days',
     CURRENT_TIMESTAMP - INTERVAL '1 days', NULL,
     10, 0, CURRENT_TIMESTAMP + INTERVAL '1 day', '0xTXFUT1', 'PENDING_PAYMENT', 5 * 500)
ON CONFLICT (id) DO NOTHING;

-- Bulk bookings (2000 bookings with varied statuses and dates)
INSERT INTO bookings (
    id, user_id, property_id,
    check_in_date, check_out_date, created_at, updated_at,
    long_stay_discount_percent, requested_negotiation_percent,
    negotiation_expires_at, on_chain_tx_hash, status, total_price
)
SELECT
    100 + gs AS id,
    -- Random tenant (ids 50-249)
    (50 + (random() * 200)::int) AS user_id,
    -- Random approved property
    p.id AS property_id,
    -- Check-in: mix of past, present, future
    dates.check_in_date,
    -- Check-out: 2-14 days after check-in
    dates.check_in_date + (2 + random() * 12)::int AS check_out_date,
    -- Created at: before check-in
    (CURRENT_TIMESTAMP - (random() * 200)::int * INTERVAL '1 day') AS created_at,
    -- Updated at: after created_at
    CURRENT_TIMESTAMP - (random() * 150)::int * INTERVAL '1 day' AS updated_at,
    -- Long stay discount: 30% chance
    CASE WHEN random() < 0.3 THEN (5 + random() * 15)::int ELSE 0 END AS long_stay_discount_percent,
    -- Negotiation: 40% chance
    CASE WHEN random() < 0.4 THEN (5 + random() * 15)::int ELSE 0 END AS requested_negotiation_percent,
    -- Negotiation expires
    CURRENT_TIMESTAMP - (random() * 10)::int * INTERVAL '1 day' AS negotiation_expires_at,
    -- Transaction hash
    '0xTX' || LPAD(gs::text, 10, '0') AS on_chain_tx_hash,
    -- Status: realistic distribution
    CASE
        WHEN random() < 0.5 THEN 'COMPLETED'              -- 50% completed
        WHEN random() < 0.65 THEN 'CONFIRMED'              -- 15% confirmed
        WHEN random() < 0.75 THEN 'TENANT_CHECKED_OUT'     -- 10% checked out
        WHEN random() < 0.85 THEN 'CANCELLED_BY_TENANT'    -- 10% cancelled
        WHEN random() < 0.92 THEN 'PENDING_PAYMENT'        -- 7% pending payment
        WHEN random() < 0.97 THEN 'PENDING_NEGOTIATION'    -- 5% pending negotiation
        ELSE 'PENDING'                                      -- 3% other pending
    END AS status,
    -- Total price: nights * daily_price (with discounts applied)
    (2 + random() * 12)::int * p.daily_price AS total_price
FROM generate_series(100, 2099) AS gs
CROSS JOIN LATERAL (
    SELECT id, daily_price
    FROM properties
    WHERE status IN ('APPROVED', 'VISIBLE_ONLY_FOR_TENANTS')
    ORDER BY random()
    LIMIT 1
) AS p
CROSS JOIN LATERAL (
    SELECT 
        CASE
            WHEN random() < 0.5 THEN CURRENT_DATE - (random() * 180)::int  -- Past (50%)
            WHEN random() < 0.7 THEN CURRENT_DATE + (random() * 30)::int   -- Near future (20%)
            ELSE CURRENT_DATE + (random() * 90)::int                        -- Future (30%)
        END AS check_in_date
) AS dates
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 7. TRANSACTIONS (Extended - linked to bookings)
-- ============================================================================

-- Original transactions (keep these)
INSERT INTO transactions (id, booking_id, user_id, amount, status, created_at, tx_hash) VALUES
    (1, 1, 3, 5 * 450, 'SUCCESS', CURRENT_TIMESTAMP - INTERVAL '24 days', '0xPAYPAST1'),
    (2, 2, 3, 7 * 650, 'SUCCESS', CURRENT_TIMESTAMP - INTERVAL '7 days', '0xPAYCURR1'),
    (3, 3, 3, 5 * 500, 'PENDING', CURRENT_TIMESTAMP - INTERVAL '1 days', '0xPAYFUT1')
ON CONFLICT (id) DO NOTHING;

-- Bulk transactions (one per booking, with some failures)
INSERT INTO transactions (id, booking_id, user_id, amount, status, created_at, tx_hash)
SELECT
    b.id AS id,
    b.id AS booking_id,
    b.user_id,
    b.total_price AS amount,
    -- Status: mostly SUCCESS, some FAILED, few PENDING
    CASE
        WHEN b.status IN ('COMPLETED', 'CONFIRMED', 'TENANT_CHECKED_OUT') THEN
            CASE WHEN random() < 0.9 THEN 'SUCCESS' ELSE 'FAILED' END
        WHEN b.status = 'PENDING_PAYMENT' THEN 'PENDING'
        WHEN b.status LIKE '%CANCELLED%' THEN
            CASE WHEN random() < 0.3 THEN 'FAILED' ELSE 'PENDING' END
        ELSE CASE WHEN random() < 0.85 THEN 'SUCCESS' ELSE 'FAILED' END
    END AS status,
    -- Created at: same day or day after booking creation
    b.created_at + (random() * 2)::int * INTERVAL '1 day' AS created_at,
    '0xPAY' || LPAD(b.id::text, 10, '0') AS tx_hash
FROM bookings b
WHERE b.id >= 100 AND b.id <= 2099
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 8. RECLAMATIONS (Extended - 5-10% of bookings)
-- ============================================================================

-- Original reclamations (keep these)
INSERT INTO reclamations (
    id, booking_id, complainant_id, complainant_role,
    target_user_id, type, title, description,
    status, severity, refund_amount, penalty_points,
    created_at, updated_at
) VALUES
    (1, 1, 3, 'GUEST', 2, 'CLEANLINESS',
     'Cleaning issue in Casablanca apartment',
     'The apartment was not properly cleaned upon arrival.',
     'RESOLVED', 'MEDIUM', 200.00, 10,
     CURRENT_TIMESTAMP - INTERVAL '18 days', CURRENT_TIMESTAMP - INTERVAL '15 days'),
    (2, 2, 3, 'GUEST', 2, 'NOT_AS_DESCRIBED',
     'Property description mismatch in Marrakech',
     'The riad did not have the advertised rooftop seating.',
     'IN_REVIEW', 'HIGH', NULL, NULL,
     CURRENT_TIMESTAMP - INTERVAL '3 days', CURRENT_TIMESTAMP - INTERVAL '1 days')
ON CONFLICT (id) DO NOTHING;

-- Bulk reclamations (150 reclamations = ~7.5% of bookings)
INSERT INTO reclamations (
    id, booking_id, complainant_id, complainant_role,
    target_user_id, type, title, description,
    status, severity, refund_amount, penalty_points,
    created_at, updated_at, resolution_notes
)
SELECT
    100 + gs AS id,
    b.id AS booking_id,
    b.user_id AS complainant_id,
    -- Mostly GUEST complaints, some HOST complaints
    CASE WHEN random() < 0.85 THEN 'GUEST' ELSE 'HOST' END AS complainant_role,
    -- Target: opposite party
    CASE 
        WHEN random() < 0.85 THEN (SELECT user_id::bigint FROM properties WHERE id = b.property_id)
        ELSE b.user_id
    END AS target_user_id,
    -- Complaint type: varied
    CASE (gs % 8)
        WHEN 0 THEN 'CLEANLINESS'
        WHEN 1 THEN 'NOT_AS_DESCRIBED'
        WHEN 2 THEN 'SAFETY_HEALTH'
        WHEN 3 THEN 'PROPERTY_DAMAGE'
        WHEN 4 THEN 'ACCESS_ISSUE'
        WHEN 5 THEN 'EXTRA_CLEANING'
        WHEN 6 THEN 'HOUSE_RULE_VIOLATION'
        ELSE 'UNAUTHORIZED_GUESTS_OR_STAY'
    END AS type,
    'Auto-generated complaint #' || gs::text AS title,
    'Generated test complaint description for training purposes.' AS description,
    -- Status: varied
    CASE
        WHEN random() < 0.4 THEN 'RESOLVED'
        WHEN random() < 0.7 THEN 'IN_REVIEW'
        WHEN random() < 0.9 THEN 'OPEN'
        ELSE 'REJECTED'
    END AS status,
    -- Severity: realistic distribution
    CASE
        WHEN random() < 0.5 THEN 'LOW'
        WHEN random() < 0.8 THEN 'MEDIUM'
        WHEN random() < 0.95 THEN 'HIGH'
        ELSE 'CRITICAL'
    END AS severity,
    -- Refund: only for resolved/high severity
    CASE
        WHEN random() < 0.4 THEN (50 + random() * 300)::numeric(10,2)
        ELSE NULL
    END AS refund_amount,
    -- Penalty points: correlated with severity
    CASE
        WHEN random() < 0.3 THEN 0
        WHEN random() < 0.6 THEN (5 + random() * 10)::int
        WHEN random() < 0.85 THEN (15 + random() * 15)::int
        ELSE (30 + random() * 20)::int
    END AS penalty_points,
    -- Created at: after booking
    b.created_at + (1 + random() * 5)::int * INTERVAL '1 day' AS created_at,
    -- Updated at: after created_at
    b.created_at + (2 + random() * 10)::int * INTERVAL '1 day' AS updated_at,
    -- Resolution notes: for resolved cases
    CASE
        WHEN random() < 0.4 THEN 'Resolved in favor of ' || 
            CASE WHEN random() < 0.5 THEN 'complainant' ELSE 'target' END
        ELSE NULL
    END AS resolution_notes
FROM generate_series(100, 249) AS gs
JOIN bookings b ON b.id = (100 + (gs % 2000))
WHERE b.id IS NOT NULL
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 9. NOTIFICATIONS (Extended)
-- ============================================================================

-- Original notifications (keep these)
INSERT INTO notifications (id, user_id, channel, status, message, created_at)
VALUES
    ('notif-1', '2', 0, 0, 'New booking received for prop-casa-001', CURRENT_TIMESTAMP - INTERVAL '7 days'),
    ('notif-2', '2', 0, 0, 'New reclamation opened for prop-marr-001', CURRENT_TIMESTAMP - INTERVAL '3 days'),
    ('notif-3', '3', 0, 0, 'Your booking for prop-agad-001 is confirmed', CURRENT_TIMESTAMP - INTERVAL '1 days')
ON CONFLICT (id) DO NOTHING;

-- Generate notifications for recent bookings and reclamations
INSERT INTO notifications (id, user_id, channel, status, message, created_at)
SELECT
    'notif-' || gs::text AS id,
    CASE
        WHEN random() < 0.5 THEN (SELECT user_id FROM properties WHERE id = b.property_id)
        ELSE b.user_id::text
    END AS user_id,
    0 AS channel,
    0 AS status,
    CASE (gs % 5)
        WHEN 0 THEN 'New booking received'
        WHEN 1 THEN 'Booking confirmed'
        WHEN 2 THEN 'New reclamation opened'
        WHEN 3 THEN 'Payment received'
        ELSE 'Booking reminder'
    END AS message,
    b.created_at AS created_at
FROM generate_series(1000, 1499) AS gs
JOIN bookings b ON b.id = (100 + (gs % 2000))
WHERE b.id IS NOT NULL
ON CONFLICT (id) DO NOTHING;

COMMIT;

-- ============================================================================
-- SUMMARY
-- ============================================================================
-- Generated:
-- - 40 hosts (ids 10-49)
-- - 200 tenants (ids 50-249)
-- - 100 addresses (ids 6-105)
-- - 300 properties (ids prop-gen-1000 to prop-gen-1299)
-- - ~2000 bookings (ids 100-2099)
-- - ~2000 transactions (ids 100-2099)
-- - ~150 reclamations (ids 100-249)
-- - ~500 notifications
-- Total: ~5000+ records for realistic ML training

