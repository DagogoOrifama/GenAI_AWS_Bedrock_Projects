import boto3
import botocore.config
import json
import base64
from datetime import datetime
from email import message_from_bytes


def extract_text_from_multipart(data):
    msg = message_from_bytes(data)
    text_content = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text_content += part.get_payload(decode=True).decode('utf-8') + "\n"
    else:
        if msg.get_content_type() == "text/plain":
            text_content = msg.get_payload(decode=True).decode('utf-8')

    return text_content.strip() if text_content else None


def generate_summary_from_bedrock(content: str) -> str:
    prompt_text = f"""Human: Summarize the following meeting notes: {content}
    Assistant:"""

    body = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 5000,
        "temperature": 0.1,
        "top_k": 250,
        "top_p": 0.2,
        "stop_sequences": ["\n\nHuman:"]
    }

    try:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3})
        )
        response = bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-v2"
        )
        response_content = response['body'].read().decode('utf-8')
        response_data = json.loads(response_content)
        return response_data.get("completion", "").strip()
    except Exception as error:
        print(f"Error generating the summary: {error}")
        return ""


def save_summary_to_s3_bucket(summary: str, bucket_name: str, object_key: str):
    s3_client = boto3.client('s3')

    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=summary)
        print("Summary saved to s3")
    except Exception as error:
        print(f"Error when saving the summary to s3: {error}")


def lambda_handler(event, context):
    decoded_body = base64.b64decode(event['body'])
    text_content = extract_text_from_multipart(decoded_body)

    if not text_content:
        return {
            'statusCode': 400,
            'body': json.dumps("Failed to extract content")
        }

    summary = generate_summary_from_bedrock(text_content)

    if summary:
        timestamp = datetime.now().strftime('%H%M%S')
        s3_key = f'summary-output/{timestamp}.txt'
        s3_bucket = 'bedrock-course-bucket'

        save_summary_to_s3_bucket(summary, s3_bucket, s3_key)
    else:
        print("No summary was generated")

    return {
        'statusCode': 200,
        'body': json.dumps("Summary generation finished")
    }
