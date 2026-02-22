from marshmallow import Schema, fields, validate


class UserRegistrationSchema(Schema):
    full_name = fields.Str(required=True, validate=validate.Length(min=1))
    phone_number = fields.Str(required=True, validate=validate.Length(min=10, max=10))
    email = fields.Email(required=True)  # Validates email format
    password = fields.Str(required=True, validate=[validate.Length(min=6)])


class UserLoginSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=1))
