from twilio.rest import Client
def sms():
  account_sid = str(open("Sercet Stuff - Raami DONT PUSH TO GIT HUB\TWILIO_act_sid.txt", "r").read()),
  auth_token = str(open("Sercet Stuff - Raami DONT PUSH TO GIT HUB\TWILIO_auth.txt", "r").read()),
  client = Client(account_sid, auth_token)

  message = client.messages.create(
    from_='+18339855157',
    body='https://sb-project-files.s3.amazonaws.com/gun_image.jpg',
    to='+14693330263'
  )

  print(message.sid)
