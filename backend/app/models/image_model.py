import bson
from datetime import datetime

class Image:
    def __init__(self, image_data, filename, image_id, cloth_type, color, season, category, upload_date=None):
        self.image_data = image_data
        self.filename = filename
        self.image_id = image_id
        self.cloth_type = cloth_type
        self.color = color
        self.season = season
        self.category = category
        self.upload_date = upload_date if upload_date else datetime.utcnow()

    def to_dict(self):
        return {
            "image_data": self.image_data,
            "filename": self.filename,
            "image_id": self.image_id,
            "cloth_type": self.cloth_type,
            "color": self.color,
            "season": self.season,
            "category": self.category,
            "upload_date": self.upload_date,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            image_data=data.get("image_data"),
            filename=data.get("filename"),
            image_id=data.get("image_id"),
            cloth_type=data.get("cloth_type"),
            color=data.get("color"),
            season=data.get("season"),
            category=data.get("category"),
            upload_date=data.get("upload_date"),
        )
