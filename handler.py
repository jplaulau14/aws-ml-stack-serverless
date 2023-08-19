import json
import logging
import boto3
import base64
import requests
from trp import Document
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

textract = boto3.client('textract')
comprehend = boto3.client('comprehend')
polly = boto3.client('polly')
transcribe = boto3.client('transcribe')
rekognition = boto3.client('rekognition')

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

def comprehend_sentiment_handler(text):
    try:
        response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
            },
            "body": json.dumps({
                "Sentiment": response['Sentiment'],
                "SentimentScore": response['SentimentScore']
            })
        }
    except Exception as e:
        logger.error(f"Error processing sentiment analysis: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to analyze sentiment: {str(e)}"})
        }

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
    
def textract_comprehend_handler(event, context):
    # Extract text from document using Textract
    textract_response = textract_handler(event, context)
    
    if textract_response["statusCode"] != 200:
        # Return error from textract_handler if there was any
        return textract_response
    
    extracted_text = " ".join(json.loads(textract_response["body"]))
    
    # Use extracted text for sentiment analysis using Comprehend
    sentiment_response = comprehend_sentiment_handler(extracted_text)
    
    # Combining the two results: extracted text and sentiment analysis
    combined_result = {
        "ExtractedText": extracted_text.split(" "),
        "SentimentAnalysis": json.loads(sentiment_response["body"])
    }
    
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        },
        "body": json.dumps(combined_result)
    }

def polly_handler(text, format='mp3'):
    try:
        response = polly.synthesize_speech(Text=text, OutputFormat=format, VoiceId='Joanna')
        
        audio_stream = response['AudioStream'].read()
        encoded_audio = base64.b64encode(audio_stream).decode('utf-8')
        
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
            },
            "body": json.dumps({"audio": encoded_audio})
        }
    except Exception as e:
        logger.error(f"Error using Polly: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to convert text to speech: {str(e)}"})
        }
    
def transcribe_handler(audio_file_path, language_code='en-US'):
    job_name = f"TranscribeJob-{int(time.time())}"

    # Assuming you'll use boto3 to upload the local file to an S3 bucket.
    s3 = boto3.resource('s3')
    bucket_name = 'aws-ml-stack-demo'
    s3_key = f"audio/{job_name}.mp3"
    s3.Bucket(bucket_name).upload_file(audio_file_path, s3_key)

    s3_audio_url = f"s3://{bucket_name}/{s3_key}"

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode=language_code,
        MediaFormat='mp3',
        Media={
            'MediaFileUri': s3_audio_url
        }
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(10)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        transcription_response = requests.get(transcription_url)
        transcription_text = transcription_response.json()['results']['transcripts'][0]['transcript']
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
            },
            "body": json.dumps({"transcription": transcription_text})
        }
    else:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to transcribe the audio."})
        }
    
def rekognition_handler(rekognition_type, image_data):
    try:
        if not rekognition_type:
            return {
                "statusCode": 400,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
                },
                "body": json.dumps({"error": "Missing 'rekognition_type' in the request."})
            }

        if not image_data:
            return {
                "statusCode": 400,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
                },
                "body": json.dumps({"error": "Missing 'image_data'."})
            }

        # Decoding the base64 image content
        image_bytes = base64.b64decode(image_data)

        if rekognition_type == 'label':
            response = rekognition.detect_labels(
                Image={
                    'Bytes': image_bytes
                }
            )
            labels = response.get('Labels', [])
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
                },
                "body": json.dumps({"labels": labels})
            }

        elif rekognition_type == 'detect_faces':
            response = rekognition.detect_faces(
                Image={
                    'Bytes': image_bytes
                },
                Attributes=['ALL']
            )
            faces = response.get('FaceDetails', [])
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
                },
                "body": json.dumps({"faces": faces})
            }

        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Invalid 'rekognition_type' specified: {rekognition_type}"})
            }

    except Exception as e:
        logger.error(f"Error using Rekognition: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to process the image: {str(e)}"})
        }

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    body = json.loads(event['body'])
    action = body.get('action')

    if action == 'textract':
        return textract_handler(event, context)
    
    elif action == 'comprehend':
        text = body.get('text')
        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing text for sentiment analysis."})
            }
        return comprehend_sentiment_handler(text)
    
    elif action == 'textract-comprehend':
        return textract_comprehend_handler(event, context)
    
    elif action == 'polly':
        text = body.get('text')
        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing text for text to speech conversion."})
            }
        return polly_handler(text)
    
    elif action == 'transcribe':
        audio_base64 = body.get('audio_base64')
        if not audio_base64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing audio for transcription."})
            }
        return transcribe_handler(audio_base64)
    
    elif action == 'rekognition':
        rekognition_type = body.get('rekognition_type')
        image_data = body.get('image_data')
        if not rekognition_type:
            logger.error("Missing 'rekognition_type' in the request.")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'rekognition_type' in the request."})
            }
        if not image_data:
            logger.error("Missing 'image_data'.")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'image_data'."})
            }
        logger.info(f"Rekognition type: {rekognition_type}")
        logger.info(f"Image data: {image_data}")
        return rekognition_handler(rekognition_type, image_data)

    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid action specified."})
        }