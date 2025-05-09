# alfred
Very fun very cool

## Usage

Install requirements:
```bash
pip install -r requirements.txt
```

## Challenges

- Need a computer vision model that's light enough to run on a Raspberry Pi with only 4 GB of RAM to run facial recognition.
- Need a camera to monitor the entrance of the house and send video to the Raspberry Pi for processing.
- Need a way to send notifications to the user when a face is recognized.
- Need a speaker to send fun messages to the user when a face is recognized.
- keep the code modular so that we can add more features later.
- use github actions and pages to handle deployment
- single user webpage with only one password field
- store password in .env
- use the local raspberry pi as a server to handle updating the face recognition model vectorized database


## Architecture

- RAG based facial recognition system
- Whenever an unfamiliar face is detected, the system will send a notification to the website on github pages
- The website will have a password field to allow the user to view the pictures and the recognized faces

## More Architecture

- Create vectorized database of faces to recognize
- Monitor camera for these faces
- SFace using ONNX for facial recognition
