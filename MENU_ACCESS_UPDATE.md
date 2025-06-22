# Menu Access Update

## Overview
Updated the role-based access control to allow users with 'user' role to access more features in the Mobile JKN Sentiment Analysis application.

## Changes Made

### Before Update
- **Admin Role**: Full access to all features
- **User Role**: Only Sentiment Prediction

### After Update
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

## Technical Changes

### Files Modified

1. **`app/main.py`**
   - Updated `render_sidebar()` function to include Data Input and Data Overview for user role
   - Updated `render_navbar_compact()` function to show navigation for user role
   - Added role-based access control in main routing
   - Set default page to 'Data Input' for both admin and user roles

### Access Control Implementation

```python
# Sidebar menu items
if st.session_state.role == 'admin':
    menu_items = [
        ('', 'Data Input'),
        ('', 'Data Overview'),
        ('', 'Model Performance'),
        ('', 'Sentiment Prediction'),
        ('', 'Word Cloud')
    ]
else:
    menu_items = [
        ('', 'Data Input'),
        ('', 'Data Overview'),
        ('', 'Sentiment Prediction')
    ]

# Page routing with role checks
elif page == 'Model Performance':
    if st.session_state.role == 'admin':
        dashboard.render_model_performance()
    else:
        st.error("Access denied. Admin privileges required.")
```

## Benefits

1. **Enhanced User Experience**: Regular users can now access more features
2. **Data Management**: Users can upload and view data
3. **Better Workflow**: Users can input data, view overview, and make predictions
4. **Maintained Security**: Admin-only features remain protected

## User Workflow

### For Regular Users
1. **Data Input**: Upload CSV files or use database data
2. **Data Overview**: Analyze data statistics and distributions
3. **Sentiment Prediction**: Perform sentiment analysis on text

### For Administrators
1. All user features plus:
2. **Model Performance**: View model metrics and performance
3. **Word Cloud**: Generate word cloud visualizations

## Testing

To test the new access control:

1. **Login as Admin**:
   - Username: `admin`
   - Password: `adminpass`
   - Should see all 5 menu items

2. **Login as User**:
   - Register a new account or use existing user account
   - Should see only 3 menu items (Data Input, Data Overview, Sentiment Prediction)

3. **Verify Access**:
   - Try accessing admin-only pages as user (should show access denied)
   - Verify all user-accessible pages work correctly

## Security Notes

- Admin-only features are protected at both UI and routing levels
- Role checks are performed before rendering sensitive pages
- Session state maintains user role information securely
- Database-level role enforcement remains intact 