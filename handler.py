import json
import logging
import boto3
import base64
import requests
from trp import Document

logger = logging.getLogger()
logger.setLevel(logging.INFO)

textract = boto3.client('textract')

def fetch_file_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error fetching content from URL: {str(e)}")
        return None
    
def clean_extracted_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Remove lines that are just URLs
        if not (line.startswith('http://') or line.startswith('https://')):
            # Remove lines with fewer than 5 characters (change this threshold as needed)
            if len(line) > 4:
                cleaned_lines.append(line.strip())  # Remove any leading/trailing whitespace

    return "\n".join(cleaned_lines)

def textract_handler(event, context):
    try:
        body = json.loads(event['body'])
        source_type = body['source_type']
        if not source_type:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing or invalid 'source_type' in the request."})
            }
        file_content_data = body['file_content']

        if source_type == 'url':
            file_content = fetch_file_from_url(file_content_data)
            if file_content is None:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "Unable to fetch content from the provided URL."})
                }
        elif source_type == 'upload':
            file_content = base64.b64decode(file_content_data)
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid source_type"})
            }

        response = textract.detect_document_text(
            Document={
                'Bytes': file_content
            }
        )
        
        results = []
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                results.append(item["Text"])
        cleaned_results = clean_extracted_text("\n".join(results))

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
            },
            "body": json.dumps(cleaned_results.split("\n"))
        }
    
    except Exception as e:
        logger.error(f"Error processing the document: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to process the document: {str(e)}"})
        }