from werkzeug.security import generate_password_hash, check_password_hash
from app.models.user_model import User
from flask import current_app as app


class AuthService:
    @staticmethod
    def register_user(full_name, phone_number, email, password):
        # Check if user already exists
        users_collection = app.mongo_db['user']

        # Find user by email
        user_exists = users_collection.find_one({"email": email}) is not None
        if user_exists:
            return {"message": "User already exists."}, False

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create and save the new user

        new_user = User(
            full_name=full_name,
            phone_number=phone_number,
            email=email,
            password_hash=hashed_password
        )

        # Insert the new user into the database
        result = users_collection.insert_one(new_user.to_dict())

        # Return the inserted user's ID
        return {'message': str(result.inserted_id)}, True

    @staticmethod
    def login_user(email, password):
        # Find user by username
        users_collection = app.mongo_db['user']

        # Find user by email
        user_data = users_collection.find_one({"email": email})

        if user_data:
            user = User.from_dict(user_data)
            if check_password_hash(user.password_hash, password):
                # Authentication successful
                return {"message": "Login successful."}, user.full_name, True

        return {"message": "Invalid email or password."}, None, False
