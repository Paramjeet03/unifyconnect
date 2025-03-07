# Hand Gesture Recognition Application

A real-time hand gesture recognition application that uses TensorFlow.js for sign language recognition and translation. The application consists of a FastAPI backend for model inference and a React frontend for the user interface.

## Features

- Real-time hand gesture recognition
- Support for multiple sign language gestures
- Confidence score for each prediction
- Modern and responsive UI
- Easy-to-use camera controls

## Prerequisites

- Python 3.9 or higher
- Node.js 14 or higher
- npm or yarn
- Webcam

## Project Structure

```
test-for-hackathon/
├── ai/                    # AI model and training code
│   └── ai/
│       └── models/       # Trained model files
├── client/               # React frontend
│   ├── src/
│   │   ├── js/
│   │   └── css/
│   └── public/
└── fastapi_backend/      # FastAPI backend
    ├── main.py
    ├── requirements.txt
    └── Dockerfile
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd fastapi_backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd client
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

## Deployment

The application can be deployed on Render:

1. Backend Deployment:
   - Create a new Web Service
   - Select the Docker environment
   - Set the root directory to `fastapi_backend`
   - Deploy

2. Frontend Deployment:
   - Create a new Static Site
   - Set the root directory to `client`
   - Set build command to `npm install && npm run build`
   - Set publish directory to `build`
   - Deploy

## API Endpoints

- `GET /`: Health check endpoint
- `POST /process_frame`: Process a video frame and return gesture prediction
- `GET /health`: Health check endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
