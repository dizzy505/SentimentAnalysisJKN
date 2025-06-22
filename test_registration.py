#!/usr/bin/env python3
"""
Test script for user registration functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.database import create_db_connection, register_user, authenticate_user

def test_registration():
    """Test the user registration system"""
    print("Testing User Registration System...")
    
    # Create database connection
    connection = create_db_connection()
    if not connection or not connection.is_connected():
        print("Failed to connect to database")
        return False
    
    print("Database connection successful")
    
    # Test 1: Register a new user
    print("\n1. Testing user registration...")
    success, message = register_user(connection, "testuser", "testpass123", "test@example.com")
    if success:
        print(f"{message}")
    else:
        print(f"{message}")
    
    # Test 2: Try to register the same user again (should fail)
    print("\n2. Testing duplicate username registration...")
    success, message = register_user(connection, "testuser", "testpass123", "test@example.com")
    if not success:
        print(f"{message} (expected behavior)")
    else:
        print(f"Should have failed: {message}")
    
    # Test 3: Authenticate the registered user
    print("\n3. Testing user authentication...")
    success, user_data = authenticate_user(connection, "testuser", "testpass123")
    if success:
        print(f"Authentication successful for user: {user_data['username']} (role: {user_data['role']})")
    else:
        print("Authentication failed")
    
    # Test 4: Try to authenticate with wrong password
    print("\n4. Testing authentication with wrong password...")
    success, user_data = authenticate_user(connection, "testuser", "wrongpass")
    if not success:
        print("Authentication correctly failed with wrong password")
    else:
        print("Authentication should have failed")
    
    # Clean up
    print("\n5. Cleaning up test data...")
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM users WHERE username = 'testuser'")
        connection.commit()
        cursor.close()
        print("Test user removed from database")
    except Exception as e:
        print(f"Failed to clean up: {e}")
    
    connection.close()
    print("\n🎉 All tests completed!")
    return True

if __name__ == "__main__":
    test_registration() 