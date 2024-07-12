import json
import boto3
import botocore
from datetime import datetime
import base64

def lambda_handler(event, context):
    event_body = json.loads(event['body'])
    message = event_body.get('message', '')

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3})
    )

    s3_client = boto3.client('s3')

    payload = {
        "text_prompts": [{"text": message}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 100
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(payload),
        modelId='stability.stable-diffusion-xl-v0',
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response.get("body").read())
    base64_image_str = response_body["artifacts"][0].get("base64")
    image_content = base64.decodebytes(base64_image_str.encode("utf-8"))

    bucket_name = 'bedrock-course-bucket'
    timestamp = datetime.now().strftime('%H%M%S')
    s3_key = f"output-images/{timestamp}.png"

    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=image_content,
        ContentType='image/png'
    )

    return {
        'statusCode': 200,
        'body': json.dumps('Image Saved to s3')
    }
