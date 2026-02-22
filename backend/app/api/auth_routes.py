from flask import request
from flask_restx import Namespace, Resource, fields
from marshmallow import ValidationError
from app.handler.user_validation import UserRegistrationSchema, UserLoginSchema
from app.services.auth_service import AuthService

auth_ns = Namespace('auth', description='Authentication operations')

user_registration_schema = UserRegistrationSchema()
user_login_schema = UserLoginSchema()

# Define your input and output models (for Swagger documentation)
user_registration_model = auth_ns.model('RegisterUser', {
    'full_name': fields.String(required=True, description='Full name of the user'),
    'phone_number': fields.String(required=True, description='Phone number of the user'),
    'email': fields.String(required=True, description='Email address of the user'),
    'password': fields.String(required=True, description='Password for the account'),
})

user_login_model = auth_ns.model('LoginUser', {
    'email': fields.String(required=True, description='Email address of the user'),
    'password': fields.String(required=True, description='Password for the account'),
})


@auth_ns.route('/register')
class RegisterUser(Resource):
    @auth_ns.expect(user_registration_model)
    def post(self):
        json_data = request.get_json()

        # Validate request data against schema
        try:
            data = user_registration_schema.load(json_data)
        except ValidationError as err:
            # If validation fails, return the errors
            return {'message': 'Validation errors', 'errors': err.messages}, 400

        # If validation is successful, proceed with registration
        message, success = AuthService.register_user(data['full_name'], data['phone_number'], data['email'],
                                                     data['password'])

        if success:
            return {'message': message}, 201
        else:
            return {'message': message}, 400


@auth_ns.route('/login')
class LoginUser(Resource):
    @auth_ns.expect(user_login_model)
    def post(self):
        json_data = request.get_json()

        # Validate request data against schema
        try:
            data = user_login_schema.load(json_data)
        except ValidationError as err:
            # If validation fails, return the errors
            return {'message': 'Validation errors', 'errors': err.messages}, 400

        # If validation is successful, proceed with login
        message, user_name, success = AuthService.login_user(data['email'], data['password'])
        if success:
            return {'message': message, 'user': user_name}, 200
        else:
            return {'message': message}, 401


@auth_ns.route('/logout')
class LogoutUser(Resource):
    def post(self):
        return {'message': 'Logged out successfully'}, 200
