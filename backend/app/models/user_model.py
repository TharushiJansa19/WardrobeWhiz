
class User:
    def __init__(self, full_name, phone_number, email, password_hash):
        self.full_name = full_name
        self.phone_number = phone_number
        self.email = email
        self.password_hash = password_hash

    def to_dict(self):
        return {
            "full_name": self.full_name,
            "phone_number": self.phone_number,
            "email": self.email,
            "password_hash": self.password_hash,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            full_name=data["full_name"],
            phone_number=data["phone_number"],
            email=data["email"],
            password_hash=data["password_hash"],
        )
