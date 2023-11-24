import os
import boto3

s3 = boto3.resource('s3')
s3.meta.client.upload_file('/Users/abhiramasonny/Developer/Python/shourya project/use/frame.jpg','sb-project-files','my-images/frame.jpg') 