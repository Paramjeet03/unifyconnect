import React, { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';

const HandGestureRecognition = ({ onGestureDetected }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [detectedGesture, setDetectedGesture] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setIsStreaming(true);
      startDetection();
    } catch (error) {
      console.error('Error accessing camera:', error);
      setError('Failed to access camera. Please ensure camera permissions are granted.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  const captureAndSendFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame on canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64
    const base64Frame = canvas.toDataURL('image/jpeg');

    try {
      // Send to FastAPI backend
      const response = await fetch('YOUR_RENDER_URL/process_frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ frame: base64Frame }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.sign) {
        setDetectedGesture(data.sign);
        setConfidence(data.confidence);
        onGestureDetected(data.sign, data.confidence);
      }
    } catch (error) {
      console.error('Error sending frame:', error);
      setError('Failed to process frame. Please try again.');
    }
  };

  const startDetection = () => {
    const interval = setInterval(() => {
      if (isStreaming) {
        captureAndSendFrame();
      }
    }, 1000); // Capture frame every second

    return () => clearInterval(interval);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="hand-gesture-recognition">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ display: isStreaming ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
      </div>
      
      <div className="controls">
        {!isStreaming ? (
          <button onClick={startCamera}>Start Camera</button>
        ) : (
          <button onClick={stopCamera}>Stop Camera</button>
        )}
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {detectedGesture && (
        <div className="results">
          <h3>Detected Sign: {detectedGesture}</h3>
          <p>Confidence: {(confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
};

HandGestureRecognition.propTypes = {
  onGestureDetected: PropTypes.func.isRequired,
};

export default HandGestureRecognition; 