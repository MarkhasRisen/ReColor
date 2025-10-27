"""Authentication middleware using Firebase ID tokens."""
from functools import wraps
from flask import request, jsonify, current_app
from firebase_admin import auth


def require_auth(f):
    """Decorator to require Firebase authentication for an endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        # Extract the token (format: "Bearer <token>")
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        token = parts[1]
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(token)
            
            # Add user info to request context
            request.user_id = decoded_token['uid']
            request.user_email = decoded_token.get('email')
            
            current_app.logger.info(f"Authenticated user: {request.user_id}")
            
        except auth.InvalidIdTokenError:
            return jsonify({'error': 'Invalid authentication token'}), 401
        except auth.ExpiredIdTokenError:
            return jsonify({'error': 'Authentication token expired'}), 401
        except Exception as e:
            current_app.logger.error(f"Authentication error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def optional_auth(f):
    """Decorator for optional authentication - extracts user if present but doesn't require it."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                try:
                    decoded_token = auth.verify_id_token(parts[1])
                    request.user_id = decoded_token['uid']
                    request.user_email = decoded_token.get('email')
                except:
                    pass  # Ignore auth errors for optional auth
        
        return f(*args, **kwargs)
    
    return decorated_function
