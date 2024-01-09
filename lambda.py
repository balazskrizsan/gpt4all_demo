import os
import json
import argparse
import logging
import boto3
import requests
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def generate_presigned_url(s3_client, client_method, method_parameters, expires_in):

    try:
        url = s3_client.generate_presigned_url(ClientMethod=client_method, Params=method_parameters, ExpiresIn=expires_in)
        print("Got presigned URL: %s", url)
    except ClientError:
        print("Couldn't get a presigned URL for client method '%s'.", "put_object")
        raise

    return url

def lambda_handler(event, context):
    print("__API_TEST__")
    URL = "host.docker.internal:43011/.well-known/openid-configuration"

    r = requests.get(url=URL, params={})

    print(r.json())

    print("__HANDLER__START__")
    s3_client = boto3.client("s3")
    url = generate_presigned_url(
        s3_client, "put_object", {"Bucket": "dil-test", "Key": "key"}, 1000
    )
    print("__HANDLER__END__")
    return {
        "presigned_upload_url": url
    }
