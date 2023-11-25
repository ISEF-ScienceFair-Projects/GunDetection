from twilio.rest import Client

account_sid = 'XXX'
auth_token = 'XXX'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='+18339855157',
  body='https://sb-project-files.s3.amazonaws.com/gun_image.jpg',
  to='+14693330263'
)

print(message.sid)
