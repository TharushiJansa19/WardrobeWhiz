import io

from flask import Blueprint, request, jsonify, abort, send_file
from flask_restx import Resource, Namespace, fields, reqparse
from werkzeug.datastructures import FileStorage
import os
import uuid
import shutil
from werkzeug.utils import secure_filename
from app.services.image_service import classify_single_image, get_distinct_categories, get_image_from_id, \
    get_images_by_category, get_similar_images, get_similar_images_by_text, find_matching, get_all_images, \
    get_images_by_color, get_distinct_colors

image_ns = Namespace('image', description='Image operations')

UPLOAD_FOLDER_SEG = 'public/Segmented'
UPLOAD_FOLDER = 'public/temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

parser = reqparse.RequestParser()
parser.add_argument('category', required=True, help="Category of the images")
parser.add_argument('page', type=int, required=False, default=1, help="Page number")
parser.add_argument('per_page', type=int, required=False, choices=[10, 20, 50], default=10, help="Images per page")

color_parser = reqparse.RequestParser()
color_parser.add_argument('color', required=True, help="Color of the images")
color_parser.add_argument('page', type=int, required=False, default=1, help="Page number")
color_parser.add_argument('per_page', type=int, required=False, choices=[10, 20, 50], default=10, help="Images per page")

get_all_images_parser = reqparse.RequestParser()
get_all_images_parser.add_argument('page', type=int, required=False, default=1, help="Page number")
get_all_images_parser.add_argument('per_page', type=int, required=False, choices=[10, 20, 50], default=10,
                                   help="Images per page")

image_by_text_parser = reqparse.RequestParser()
image_by_text_parser.add_argument('text', required=True, help="Text to get images")

get_image_parser = reqparse.RequestParser()
get_image_parser.add_argument('image_id', required=True, help="image id")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


upload_parser = reqparse.RequestParser()
upload_parser.add_argument('image', location='files',
                           type=FileStorage, required=True,
                           help='Image file')

classify_model = image_ns.model('Image Classify Model', {
    'message': fields.String(description='Read image meta data from LLM and store image and embeddings'),
})

categories_model = image_ns.model('CategoriesModel', {
    'categories': fields.List(fields.String, description="List of distinct image categories")
})

colors_model = image_ns.model('ColorsModel', {
    'colors': fields.List(fields.String, description="List of distinct image categories")
})

image_info_model = image_ns.model('ImageInfoModel', {
    'page': fields.Integer(description="Current page number"),
    'per_page': fields.Integer(description="Number of images per page"),
    'total': fields.Integer(description="Total number of images in the category"),
    'images': fields.List(fields.String(description="Image data or references"))
})

similar_images_model = image_ns.model('SimilarImagesModel', {
    'message': fields.String(description="Operation outcome message"),
    'data': fields.List(fields.String(description="References or data of similar images"))
})

similar_images_by_text_model = image_ns.model('SimilarImagesByTextModel', {
    'message': fields.String(description="Operation outcome message"),
    'data': fields.List(fields.String(description="References or data of similar images"))
})

image_model = image_ns.model('ImageModel', {
    'message': fields.String(description='Get image file from image id'),
})


@image_ns.route('/')
class ImagesByCategory(Resource):
    @image_ns.expect(get_all_images_parser)
    @image_ns.marshal_with(image_info_model)
    def get(self):
        args = get_all_images_parser.parse_args()
        page = args['page']
        per_page = args['per_page']

        images, total = get_all_images(page, per_page)
        return {
            'category': None,
            'page': page,
            'per_page': per_page,
            'total': total,
            'images': images
        }, 200


@image_ns.route('/classify')
class ImageClassify(Resource):
    @image_ns.expect(upload_parser)
    @image_ns.response(201, 'Success', model=classify_model)  # Link the success response to the model
    @image_ns.response(400, 'Validation Error')
    def post(self):
        args = upload_parser.parse_args()
        file = args['image']  # Using the parser to access the uploaded file

        if file.filename == '' or not allowed_file(file.filename):
            return {'message': 'No image selected for uploading or file type not allowed'}, 400

        # Generate a unique filename using uuid
        ext = file.filename.rsplit('.', 1)[1].lower()
        id = uuid.uuid4()
        filename = secure_filename(f"{id}.{ext}")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_path_seg = os.path.join(UPLOAD_FOLDER_SEG, filename)

        # It's a good practice to use a context manager for file operations
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)
        file.seek(0)  # Reset the file's stream position
        with open(file_path_seg, 'wb') as f:
            shutil.copyfileobj(file, f)

        try:
            # Process the image to classify, save in MongoDB, and store embeddings in Pinecone
            message, success = classify_single_image(file_path, file_path_seg, id)
        finally:
            # Ensure the temporary file is deleted even if an error occurs during processing
            os.remove(file_path)
            os.remove(file_path_seg)

        if success:
            return {'message': message}, 201
        else:
            return {'message': message}, 400


@image_ns.route('/categories')
class ImageCategories(Resource):
    @image_ns.marshal_with(categories_model)
    def get(self):
        categories = get_distinct_categories()
        return {'categories': categories}, 200


@image_ns.route('/images_by_category')
class ImagesByCategory(Resource):
    @image_ns.expect(parser)
    @image_ns.marshal_with(image_info_model)
    def get(self):
        """Return images for a given category with pagination."""
        args = parser.parse_args()
        category = args['category']
        page = args['page']
        per_page = args['per_page']

        images, total = get_images_by_category(category, page, per_page)
        return {
            'page': page,
            'per_page': per_page,
            'total': total,
            'images': images
        }, 200


@image_ns.route('/colors')
class ImageCategories(Resource):
    @image_ns.marshal_with(colors_model)
    def get(self):
        colors = get_distinct_colors()
        return {'colors': colors}, 200


@image_ns.route('/images_by_colors')
class ImagesByCategory(Resource):
    @image_ns.expect(color_parser)
    @image_ns.marshal_with(image_info_model)
    def get(self):
        """Return images for a given category with pagination."""
        args = color_parser.parse_args()
        color = args['color']
        page = args['page']
        per_page = args['per_page']

        images, total = get_images_by_color(color, page, per_page)
        return {
            'page': page,
            'per_page': per_page,
            'total': total,
            'images': images
        }, 200


@image_ns.route('/get_image')
class GetImage(Resource):
    # < img src = "/path/to/serve_image_base64_endpoint" >
    @image_ns.expect(get_image_parser)
    def get(self):
        args = get_image_parser.parse_args()
        image_id = args['image_id']
        image_record = get_image_from_id(image_id)

        if not image_record:
            abort(404)  # Not Found if the image doesn't exist

        # Convert the binary image data to a BytesIO object
        image_bytes = io.BytesIO(image_record['image_data'])
        # Use send_file to send the image data as a file response
        return send_file(image_bytes, mimetype='image/jpeg')


@image_ns.route('/find_similar')
class ImageClassify(Resource):
    @image_ns.expect(upload_parser)
    @image_ns.marshal_with(similar_images_model, code=200, description="Success")
    @image_ns.doc(responses={400: 'Validation Error'})
    def post(self):
        args = upload_parser.parse_args()
        file = args['image']  # Using the parser to access the uploaded file

        if file.filename == '' or not allowed_file(file.filename):
            return {'message': 'No image selected for uploading or file type not allowed'}, 400

        # Generate a unique filename using uuid
        ext = file.filename.rsplit('.', 1)[1].lower()
        id = uuid.uuid4()
        filename = secure_filename(f"{id}.{ext}")
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # It's a good practice to use a context manager for file operations
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)  # More secure file saving

        success = True
        err = None

        try:
            # Process the image to classify, save in MongoDB, and store embeddings in Pinecone
            result = get_similar_images(file_path, id)
        except Exception as e:
            print(e)
            success = False
            err = e
        finally:
            # Ensure the temporary file is deleted even if an error occurs during processing
            os.remove(file_path)

        if not success:
            return {'message': str(err)}, 400

        return {'message': "success", 'data': result}, 200


@image_ns.route('/find_similar_by_text')
class ImageClassifyByText(Resource):
    @image_ns.expect(image_by_text_parser)
    @image_ns.marshal_with(similar_images_model, code=200, description="Success")
    @image_ns.doc(responses={400: 'Validation Error'})
    def get(self):
        args = image_by_text_parser.parse_args()
        text = args['text']

        success = True
        err = None

        try:
            # Process the image to classify, save in MongoDB, and store embeddings in Pinecone
            result = get_similar_images_by_text(text)
        except Exception as e:
            print(e)
            success = False
            err = e

        if not success:
            return {'message': str(err)}, 400

        return {'message': "success", 'data': result}, 200


@image_ns.route('/find_matching')
class ImageMatching(Resource):
    @image_ns.expect(upload_parser)
    @image_ns.marshal_with(similar_images_model, code=200, description="Success")
    @image_ns.doc(responses={400: 'Validation Error'})
    def post(self):
        args = upload_parser.parse_args()
        file = args['image']  # Using the parser to access the uploaded file

        if file.filename == '' or not allowed_file(file.filename):
            return {'message': 'No image selected for uploading or file type not allowed'}, 400

        # Generate a unique filename using uuid
        ext = file.filename.rsplit('.', 1)[1].lower()
        id = uuid.uuid4()
        filename = secure_filename(f"{id}.{ext}")
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # It's a good practice to use a context manager for file operations
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)  # More secure file saving

        success = True
        err = None

        try:
            # Process the image to classify, save in MongoDB, and store embeddings in Pinecone
            result = find_matching(file_path)
        except Exception as e:
            print(e)
            success = False
            err = e
        finally:
            # Ensure the temporary file is deleted even if an error occurs during processing
            os.remove(file_path)

        if not success:
            return {'message': str(err)}, 400

        return {'message': "success", 'data': result}, 200
