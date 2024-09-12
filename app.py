from flask import Flask, request, jsonify
import random
import sendgrid
from sendgrid.helpers.mail import Mail
import firebase_admin
from firebase_admin import firestore

# Initialize Firebase
cred = firebase_admin.credentials.Certificate('path/to/your-firebase-service-account.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

# Function to generate a 6-digit OTP
def generate_otp():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

# Function to send OTP via email using SendGrid
def send_otp_email(recipient_email, otp_code):
    sg = sendgrid.SendGridAPIClient(api_key='your-sendgrid-api-key')
    from_email = 'your-email@example.com'
    subject = 'Your OTP Code'
    content = f'Your OTP code is {otp_code}'

    mail = Mail(
        from_email=from_email,
        to_emails=recipient_email,
        subject=subject,
        plain_text_content=content
    )

    try:
        response = sg.send(mail)
        return response.status_code == 202
    except Exception as e:
        print(e)
        return False

@app.route('/send_otp', methods=['POST'])
def send_otp():
    data = request.json
    email = data['email']
    otp = generate_otp()

    # Optionally store OTP in Firestore for verification
    db.collection('otps').document(email).set({'otp': otp})

    if send_otp_email(email, otp):
        return jsonify({'message': 'OTP sent successfully'}), 200
    else:
        return jsonify({'message': 'Failed to send OTP'}), 500

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data['email']
    otp_input = data['otp']

    # Retrieve OTP from Firestore
    otp_data = db.collection('otps').document(email).get()
    if otp_data.exists and otp_data.to_dict()['otp'] == otp_input:
        return jsonify({'message': 'OTP verified successfully'}), 200
    else:
        return jsonify({'message': 'Invalid OTP'}), 400

if __name__ == '__main__':
    app.run(debug=True)
