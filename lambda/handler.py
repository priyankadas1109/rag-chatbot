import json
import boto3
import os
from botocore.exceptions import BotoCoreError, ClientError

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query", "").strip()

        if not query:
            return {
                "statusCode": 400,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Content-Type": "application/json"
                },
                "body": json.dumps({"error": "Query is required"})
            }

        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
        model_id = os.environ.get("CLAUDE_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

        payload = {
            "prompt": query,
            "temperature": 0.0,
            "max_tokens_to_sample": 1024
        }

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        response_body = json.loads(response['body'].read())

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            },
            "body": json.dumps({"response": response_body.get("completion", "")})
        }

    except (BotoCoreError, ClientError) as aws_error:
        return {
            "statusCode": 502,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": "AWS Bedrock invocation failed", "details": str(aws_error)})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": "Internal server error", "details": str(e)})
        }
