import React from 'react';
import ReactDOM from 'react-dom/client';
import './css/hand-gesture.css';
import HandGestureRecognition from './js/components/HandGestureRecognition';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <div className="app-container">
      <h1>Hand Gesture Recognition</h1>
      <HandGestureRecognition 
        onGestureDetected={(sign, confidence) => {
          console.log(`Detected sign: ${sign} with confidence: ${confidence}`);
        }} 
      />
    </div>
  </React.StrictMode>
);
