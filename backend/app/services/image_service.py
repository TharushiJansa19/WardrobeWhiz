from bson import Binary
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.cm as cm
from flask import current_app as app
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import List
from app.models.image_model import Image
from PIL import Image as PILImage
from app.services.pinecorn_service import insert_into_pinecone, get_similar_records, get_similar_records_by_text
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from flask import current_app as app

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

prompt_template_str = """\
    Can you summarize the cloting item in the image and return a response \
    with the following JSON format: \
"""

prompt_give_matching_cloths_str = """\
    Can you give information for a matching clothing items to wear with the cloth in the image\
    If this is top wear give a matching bottom wear and vice versa.\
    with the following JSON format: \
"""


class ReceiptInfo(BaseModel):
    cloth_type: str = Field(..., description="Type of the cloth or fashion item")
    color: str = Field(..., description="color of the item")
    season: str = Field(..., description="The season which this cloth is more suitable")
    category: str = Field(..., description="Is this a women's or men's item. unisex for anything other than those two")
    summary: str = Field(
        ...,
        description="a simple description including color, type ,when can this be worn and suitable events to wear,  "
                    "etc. ",
    )


def get_nodes_from_objs(
        objs: List[ReceiptInfo], image_files: List[str], ids: List[str]
) -> TextNode:
    """Get nodes from objects."""
    nodes = []
    for image_file, obj, id in zip(image_files, objs, ids):
        node = TextNode(
            text=obj.summary,
            metadata={
                "cloth_type": obj.cloth_type,
                "color": obj.color,
                "season": obj.season,
                "category": obj.category,
                "image_file": str(image_file),
                'image_id': id
            },
            excluded_embed_metadata_keys=["image_file"],
            excluded_llm_metadata_keys=["image_file"],
        )
        nodes.append(node)
    return nodes


def pydantic_gemini(output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(
        api_key=app.config['GOOGLE_API_KEY'], model_name="models/gemini-pro-vision"
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )

    response = llm_program()
    return response


def apply_mask_image(mask, image, filename, label, save):
    # Convert PIL Image to numpy array if it's not already an array
    image = np.array(image)

    # Initialize a masked_image array of zeros with the same shape as the input image
    masked_image = np.zeros_like(image)

    # Apply the mask to the image
    # This operation copies pixels from the original image where the mask is 1
    masked_image[mask == 1] = image[mask == 1]

    if save:
        # Convert the numpy array back to a PIL Image for saving
        masked_image_pil = PILImage.fromarray(masked_image)

        # Save the masked image to a file
        masked_image_pil.save(filename)

    # Return the numpy array of the masked image
    return masked_image


def background_rem(pred_seg, image, filename):
    # Convert the tensor to a numpy array
    segmented_img_numpy = pred_seg.detach().cpu().numpy()
    # Get the unique labels
    labels = np.unique(segmented_img_numpy)

    combined_mask = np.zeros_like(segmented_img_numpy, dtype=np.uint8)
    # Loop through each label
    for label in labels:
        if label == 0:
            continue
        # Create a mask based on the label
        combined_mask[segmented_img_numpy == label] = 1
    _ = apply_mask_image(combined_mask, image, filename, "background_removed", True)


# def apply_mask_image(mask, image, filename, save):
#     # Create a masked image using the mask
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#
#     if save:
#         # Save the masked image to a file
#         cv2.imwrite(filename, masked_image)
#
#     return masked_image
#
#
# def background_rem(image_path, filename, save=True):
#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image could not be read.")
#
#     # Convert to HSV color space to simplify color-based segmentation
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     # Define range for background colors (adjust these values based on your needs)
#     lower_val = np.array([0, 0, 120])  # Example: low brightness values
#     upper_val = np.array([180, 255, 255])
#
#     # Create a mask that captures areas not in the background range
#     mask = cv2.inRange(hsv, lower_val, upper_val)
#     mask = 255 - mask  # Invert mask if necessary
#
#     # Apply mask and save or return image
#     result = apply_mask_image(mask, image, filename, save)
#     return result


def model_in(image):
    # image = Image.open(filename)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    return pred_seg


def aprocess_image_file(image_file):
    # # should load one file
    image = PILImage.open(image_file)
    #
    # # Example of further processing - replace with your actual logic
    img_dir = str(image_file)
    #
    pred_seg = model_in(image)
    background_rem(pred_seg, image, img_dir)

    img_docs = SimpleDirectoryReader(input_files=[img_dir]).load_data()
    output = pydantic_gemini(ReceiptInfo, img_docs, prompt_template_str)
    return output


def aprocess_image_file_matching(image_file):
    # # should load one file
    image = PILImage.open(image_file)
    #
    # # Example of further processing - replace with your actual logic
    img_dir = str(image_file)
    #
    pred_seg = model_in(image)
    background_rem(pred_seg, image, img_dir)

    img_docs = SimpleDirectoryReader(input_files=[img_dir]).load_data()
    output = pydantic_gemini(ReceiptInfo, img_docs, prompt_give_matching_cloths_str)
    return output


def get_embeddings_from_model(img_dir, id):
    output = aprocess_image_file(img_dir)
    # add the description to the index
    nodes = get_nodes_from_objs([output], [img_dir], [str(id)])
    return nodes, output


def classify_single_image(image_path, image_path_seg, id):
    # Replace these with actual calls to your model API
    nodes, output = get_embeddings_from_model(image_path_seg, id)

    # Save the image in MongoDB and get the document ID
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    filename = os.path.basename(image_path)
    image = Image(
        image_data=image_data,
        filename=filename,
        image_id=str(id),
        cloth_type=output.cloth_type,
        color=output.color,
        season=output.season,
        category=output.category
    )

    images_collection = app.mongo_db['image']

    image_dict = image.to_dict()
    image_dict['image_data'] = Binary(image_dict['image_data'])

    images_collection.insert_one(image_dict)

    # Store the embeddings and document ID in Pinecone
    insert_into_pinecone(nodes)

    return 'Image processed and data stored', True


def get_distinct_categories():
    try:
        images_collection = app.mongo_db['image']
        # Define an aggregation pipeline to transform and collect distinct categories
        pipeline = [
            {"$project": {
                "cloth_types": {
                    "$split": ["$cloth_type", ", "]  # Split the category field into an array of categories
                }
            }},
            {"$unwind": "$cloth_types"},  # Deconstruct the array
            {"$group": {
                "_id": {"$toLower": "$cloth_types"}  # Convert categories to lowercase and group by them to get distinct
            }},
            {"$sort": {"_id": 1}}  # Sort alphabetically by category
        ]
        result = images_collection.aggregate(pipeline)
        # Extract categories from the aggregation result
        distinct_categories = [doc['_id'] for doc in result]
        return distinct_categories
    except Exception as e:
        print(f"An error occurred while fetching categories: {e}")
        return []


def get_distinct_colors():
    try:
        images_collection = app.mongo_db['image']
        # Define an aggregation pipeline to transform and collect distinct colors
        pipeline = [
            {"$project": {
                "colors": {
                    "$split": ["$color", ", "]  # Split the color field into an array of colors
                }
            }},
            {"$unwind": "$colors"},  # Deconstruct the array
            {"$group": {
                "_id": {"$toLower": "$colors"}  # Convert colors to lowercase and group by them to get distinct
            }},
            {"$sort": {"_id": 1}}  # Sort alphabetically by color
        ]
        result = images_collection.aggregate(pipeline)
        # Extract colors from the aggregation result
        distinct_colors = [doc['_id'] for doc in result]
        return distinct_colors
    except Exception as e:
        print(f"An error occurred while fetching colors: {e}")
        return []


# def get_images_by_category(category, page, size):
#     skip = (page - 1) * size
#     query = {"category": category}
#
#     # Fetch the images
#     images_collection = app.mongo_db['image']
#     cursor = images_collection.find(query).skip(skip).limit(size)
#     images = list(cursor)
#
#     # Count the total documents to calculate total pages
#     total_documents = images_collection.count_documents(query)
#     total_pages = (total_documents + size - 1) // size
#
#     # Assuming images are dictionaries that can be directly serialized to JSON
#     # If they contain non-serializable fields (like ObjectId), convert them accordingly
#     return images, total_pages


def get_image_from_id(image_id):
    images_collection = app.mongo_db['image']
    image_record = images_collection.find_one({'image_id': image_id})

    return image_record


# def get_images_by_category(category, page, per_page):
#     skip_amount = (page - 1) * per_page
#     try:
#         query = {'cloth_type': category}
#         images_collection = app.mongo_db['image']
#         total = images_collection.count_documents(query)
#         images_cursor = images_collection.find(query).skip(skip_amount).limit(per_page)
#         images = list(images_cursor)
#
#         images = [str(image['image_id']) for image in images]
#
#         return images, total
#     except Exception as e:
#         print(f"An error occurred while fetching images: {e}")
#         return [], 0


def get_images_by_category(category, page, per_page):
    skip_amount = (page - 1) * per_page
    try:
        # Using regex to match color in a comma-separated string
        regex_pattern = f'\\b{category}\\b'  # The \b denotes a word boundary
        query = {'cloth_type': {'$regex': regex_pattern, '$options': 'i'}}
        images_collection = app.mongo_db['image']
        total = images_collection.count_documents(query)
        images_cursor = images_collection.find(query).skip(skip_amount).limit(per_page)
        images = list(images_cursor)

        images = [str(image['image_id']) for image in images]

        return images, total
    except Exception as e:
        print(f"An error occurred while fetching images: {e}")
        return [], 0


def get_images_by_color(color, page, per_page):
    skip_amount = (page - 1) * per_page
    try:
        # Using regex to match color in a comma-separated string
        regex_pattern = f'\\b{color}\\b'  # The \b denotes a word boundary
        query = {'color': {'$regex': regex_pattern, '$options': 'i'}}
        images_collection = app.mongo_db['image']
        total = images_collection.count_documents(query)
        images_cursor = images_collection.find(query).skip(skip_amount).limit(per_page)
        images = list(images_cursor)

        images = [str(image['image_id']) for image in images]

        return images, total
    except Exception as e:
        print(f"An error occurred while fetching images: {e}")
        return [], 0


def get_all_images(page, per_page):
    skip_amount = (page - 1) * per_page
    try:
        images_collection = app.mongo_db['image']
        total = images_collection.count_documents({})
        images_cursor = images_collection.find({}).skip(skip_amount).limit(per_page)
        images = list(images_cursor)

        images = [str(image['image_id']) for image in images]

        return images, total
    except Exception as e:
        print(f"An error occurred while fetching images: {e}")
        return [], 0


def get_similar_images(image_path, id):
    # Replace these with actual calls to your model API
    nodes, output = get_embeddings_from_model(image_path, id)

    # Store the embeddings and document ID in Pinecone
    results = get_similar_records(nodes)

    try:
        # Default to an empty list if 'matches' key is missing
        matches = results.get('matches', [])

        # Filter and sort the matches
        filtered_sorted_matches = sorted(
            [match for match in matches if match.get('score', 0) > 0.90],  # Use .get with a default score of 0
            key=lambda x: x['score'], reverse=True
        )

        # Extract image_id from the filtered and sorted matches, safely accessing 'metadata' and 'image_id'
        image_ids = [match['metadata'].get('image_id') for match in filtered_sorted_matches if
                     match['metadata'].get('image_id') is not None]

    except KeyError as e:
        print(f"A KeyError occurred: {e}")
        image_ids = []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        image_ids = []

    return image_ids


def get_similar_images_by_text(text):
    # Store the embeddings and document ID in Pinecone
    results = get_similar_records_by_text(text)

    try:
        # Default to an empty list if 'matches' key is missing
        matches = results.get('matches', [])

        # Filter and sort the matches
        filtered_sorted_matches = sorted(
            [match for match in matches if match.get('score', 0) > 0.75],  # Use .get with a default score of 0
            key=lambda x: x['score'], reverse=True
        )

        # Extract image_id from the filtered and sorted matches, safely accessing 'metadata' and 'image_id'
        image_ids = [match['metadata'].get('image_id') for match in filtered_sorted_matches if
                     match['metadata'].get('image_id') is not None]

    except KeyError as e:
        print(f"A KeyError occurred: {e}")
        image_ids = []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        image_ids = []

    return image_ids


def find_matching(img_dir):
    output = aprocess_image_file_matching(img_dir)
    output = output.summary
    output = output.split("would")

    results = get_similar_records_by_text(output[0])

    try:
        # Default to an empty list if 'matches' key is missing
        matches = results.get('matches', [])

        # Filter and sort the matches
        filtered_sorted_matches = sorted(
            [match for match in matches if match.get('score', 0) > 0.75],  # Use .get with a default score of 0
            key=lambda x: x['score'], reverse=True
        )

        # Extract image_id from the filtered and sorted matches, safely accessing 'metadata' and 'image_id'
        image_ids = [match['metadata'].get('image_id') for match in filtered_sorted_matches if
                     match['metadata'].get('image_id') is not None]

    except KeyError as e:
        print(f"A KeyError occurred: {e}")
        image_ids = []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        image_ids = []

    return image_ids
