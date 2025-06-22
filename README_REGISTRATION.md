# User Registration Feature

## Overview
The Mobile JKN Sentiment Analysis application now includes a comprehensive user registration and authentication system that allows users to create accounts and access the sentiment analysis features.

## Features

### 1. User Registration
- **Public Registration**: New users can create accounts through the registration form
- **Email Support**: Optional email field for user accounts
- **Password Validation**: 
  - Minimum 6 characters
  - Must contain at least one letter and one number
  - Password confirmation required
- **Username Uniqueness**: Prevents duplicate usernames
- **Email Uniqueness**: Prevents duplicate email addresses (if provided)

### 2. User Authentication
- **Database-based**: All user credentials stored securely in MySQL database
- **Password Hashing**: Passwords are hashed using SHA-256
- **Session Management**: User sessions with role-based access control

### 3. Role-Based Access Control
- **Admin Role**: Full access to all features
  - Data Input
  - Data Overview
  - Model Performance
  - Sentiment Prediction
  - Word Cloud
- **User Role**: Access to core features
  - Data Input
  - Data Overview
  - Sentiment Prediction

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    role ENUM('admin', 'user') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## Usage

### For New Users
1. Navigate to the application
2. Click on the "Register" tab
3. Fill in the registration form:
   - Username (required)
   - Email (optional)
   - Password (required, minimum 6 characters with letters and numbers)
   - Confirm Password (required)
4. Click "Register"
5. Login with your new credentials

### For Administrators
1. Login with admin credentials (default: admin/adminpass)
2. Access all features including:
   - Data Input and management
   - Data Overview and analytics
   - Model Performance monitoring
   - Sentiment Prediction
   - Word Cloud generation

### For Regular Users
1. Login with your credentials
2. Access the following features:
   - **Data Input**: Upload CSV files and manage data
   - **Data Overview**: View data statistics and visualizations
   - **Sentiment Prediction**: Analyze text sentiment
3. Limited to core functionality (no admin features)

## Security Features

- **Password Hashing**: All passwords are hashed using SHA-256
- **Input Validation**: Comprehensive validation for registration data
- **SQL Injection Prevention**: Parameterized queries for all database operations
- **Session Management**: Secure session handling with role-based access
- **Database Constraints**: Unique constraints on username and email

## Default Admin Account

The system automatically creates a default admin account:
- **Username**: admin
- **Password**: adminpass
- **Email**: admin@jkn.com
- **Role**: admin

## Testing

Run the test script to verify the registration system:
```bash
python test_registration.py
```

This will test:
- User registration
- Duplicate username prevention
- User authentication
- Wrong password handling
- Database cleanup

## File Structure

```
app/
├── database.py          # Database operations including user authentication
├── dashboard.py         # UI components including registration form
├── main.py             # Main application with navigation
├── utils.py            # Session state management
└── config.py           # Database configuration

test_registration.py    # Test script for registration functionality
README_REGISTRATION.md  # This documentation file
```

## Technical Implementation

### Key Functions

#### Database Functions (`database.py`)
- `register_user()`: Register new users
- `authenticate_user()`: Authenticate user login
- `get_user_by_username()`: Get user information

#### UI Functions (`dashboard.py`)
- `render_login()`: Login and registration forms

#### Session Management (`utils.py`)
- `init_session_state()`: Initialize user session variables

## Migration from Hardcoded Users

The system has been migrated from hardcoded user configuration to database-based user management:

### Before
```python
# config.py
users = {
    'admin': {'password': 'hash', 'role': 'admin'},
    'user': {'password': 'hash', 'role': 'user'}
}
```

### After
```sql
-- Database table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    role ENUM('admin', 'user') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Future Enhancements

Potential improvements for the registration system:
- Email verification for new accounts
- Password reset functionality
- User profile management
- Account deactivation
- Login attempt limiting
- Two-factor authentication
- Audit logging for user actions 