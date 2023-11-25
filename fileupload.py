import boto3

import boto3
session = boto3.Session(
    aws_access_key_id=str(open("Sercet Stuff - Raami DONT PUSH TO GIT HUB\AWS_access.txt", "r").read()),
    aws_secret_access_key=str(open("Sercet Stuff - Raami DONT PUSH TO GIT HUB\AWS_sec.txt", "r").read()),
)
s3 = session.resource('s3')
s3.meta.client.upload_file(Filename='gunImages\gunMan.jpg', Bucket='sb-project-files', Key='gun_image.jpg')
