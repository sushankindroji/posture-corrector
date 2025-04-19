import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { w3cwebsocket as W3CWebSocket } from 'websocket';
import Navbar from '../components/Navbar';

const PostureDetail = () => {
  const { postureId } = useParams();
  const navigate = useNavigate();
  const [isRunning, setIsRunning] = useState(false);
  const [processId, setProcessId] = useState('');
  const [score, setScore] = useState(0);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  const startAnalysis = async () => {
    try {
      const response = await fetch(`http://localhost:8000/start-posture/${postureId}`, {
        method: 'POST'
      });
      const data = await response.json();
      setProcessId(data.process_id);
      setIsRunning(true);
      
      // Connect to WebSocket
      const newWs = new W3CWebSocket(`ws://localhost:8000/ws/${data.process_id}`);
      newWs.onmessage = (message) => {
        const data = JSON.parse(message.data);
        if (data.type === 'data') {
          // Update score based on data
          if (data.score) {
            setScore(data.score);
          }
        }
      };
      setWs(newWs);
    } catch (error) {
      console.error('Error starting analysis:', error);
    }
  };

  const stopAnalysis = async () => {
    try {
      await fetch(`http://localhost:8000/stop-posture/${processId}`, {
        method: 'POST'
      });
      setIsRunning(false);
      if (ws) {
        ws.close();
      }
    } catch (error) {
      console.error('Error stopping analysis:', error);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Navbar */}
      <Navbar />
      
      <div className="max-w-7xl mx-auto pt-24 pb-16 px-4 sm:px-6 lg:px-8">
        {/* Header section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-text-primary mb-2">
            {postureId} <span className="text-primary">Analysis</span>
          </h1>
          <p className="text-text-secondary">Start your posture analysis with a single click</p>
        </div>
        
        {/* Main content */}
        <div className="grid grid-cols-1 gap-8">
          {/* Controls Panel */}
          <div className="bg-background-light rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-text-primary mb-4">Controls</h2>
            
            <div className="flex flex-col md:flex-row gap-4">
              <button
                onClick={startAnalysis}
                disabled={isRunning}
                className={`px-5 py-2 rounded-md text-white font-medium ${isRunning 
                  ? 'bg-blue-400 opacity-50 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600 transition-all duration-300'}`}
              >
                <div className="flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Start Analysis
                </div>
              </button>
              
              <button
                onClick={stopAnalysis}
                disabled={!isRunning}
                className={`px-5 py-2 rounded-md font-medium ${!isRunning 
                  ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                  : 'bg-red-500 hover:bg-red-600 text-white transition-all duration-300'}`}
              >
                <div className="flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                  </svg>
                  Stop Analysis
                </div>
              </button>
            </div>
            
            {isRunning && (
              <div className="mt-6 p-4 bg-background rounded-md border border-primary">
                <div className="flex items-center">
                  <div className="relative mr-3">
                    <div className="w-3 h-3 bg-primary rounded-full absolute animate-ping"></div>
                    <div className="w-3 h-3 bg-primary rounded-full"></div>
                  </div>
                  <span className="text-text-primary">Analysis in progress...</span>
                </div>
              </div>
            )}
            
            {!isRunning && score > 0 && (
              <div className="mt-6">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="font-medium text-text-primary">Performance Score</h3>
                  <span className="text-primary font-bold">{score}%</span>
                </div>
                <div className="w-full bg-background rounded-full h-2.5">
                  <div 
                    className="bg-primary h-2.5 rounded-full" 
                    style={{ width: `${score}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Instructions section */}
        <div className="mt-8">
          <div className="bg-background-light rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-text-primary mb-4">How to Use</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="flex flex-col">
                <div className="text-primary mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="font-medium text-text-primary mb-1">Step 1</h3>
                <p className="text-text-secondary text-sm">Click "Start Analysis" to begin posture detection</p>
              </div>
              <div className="flex flex-col">
                <div className="text-primary mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="font-medium text-text-primary mb-1">Step 2</h3>
                <p className="text-text-secondary text-sm">Maintain a proper posture during the analysis session</p>
              </div>
              <div className="flex flex-col">
                <div className="text-primary mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="font-medium text-text-primary mb-1">Step 3</h3>
                <p className="text-text-secondary text-sm">When finished, click "Stop Analysis" to see your score</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PostureDetail;