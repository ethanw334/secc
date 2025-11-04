import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './components/1-FileUpload';
import AnalysisView from './components/2-AnalysisView';
import ResultsDisplay from './components/3-ResultsDisplay';

// Define constants
const APP_STATUS = {
  IDLE: 'IDLE',
  UPLOADING: 'UPLOADING',
  PROCESSING: 'PROCESSING',
  COMPLETE: 'COMPLETE',
  ERROR: 'ERROR',
};

const API_URL = 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState(APP_STATUS.IDLE);
  const [logs, setLogs] = useState([]);
  const [findings, setFindings] = useState(null);
  const ws = useRef(null);

  // Effect to clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const addLog = (message, type = 'log') => {
    setLogs((prevLogs) => [...prevLogs, { message, type }]);
  };

  const handleAnalyzeClick = async () => {
    if (files.length === 0) {
      addLog('No files selected.', 'error');
      return;
    }

    setStatus(APP_STATUS.UPLOADING);
    addLog(`Uploading ${files.length} file(s)...`);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      // 1. HTTP Upload
      const uploadResponse = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      const { session_id } = uploadResponse.data;
      addLog('Upload complete. Connecting to analysis server...');
      setStatus(APP_STATUS.PROCESSING);

      // 2. WebSocket Connection
      ws.current = new WebSocket(`${API_URL.replace('http', 'ws')}/ws/analysis`);

      ws.current.onopen = () => {
        addLog('Connection open. Starting analysis...');
        // Send the session_id to start the backend process
        ws.current.send(JSON.stringify({ session_id: session_id }));
      };

      ws.current.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'log') {
          addLog(msg.message);
        } else if (msg.type === 'result') {
          addLog('Analysis complete! Results received.');
          setFindings(msg.data);
          setStatus(APP_STATUS.COMPLETE);
          ws.current.close();
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        addLog('A WebSocket error occurred. Check the console.', 'error');
        setStatus(APP_STATUS.ERROR);
      };

      ws.current.onclose = () => {
  addLog('Connection closed.');
  // We no longer set ERROR status here,
  // as it races with the COMPLETE status set in onmessage.
  // The onerror handler will catch actual errors.
};

    } catch (error) {
      console.error('Upload Error:', error);
      addLog(error.response?.data?.detail || 'Upload failed.', 'error');
      setStatus(APP_STATUS.ERROR);
    }
  };

  const handleReset = () => {
    setFiles([]);
    setStatus(APP_STATUS.IDLE);
    setLogs([]);
    setFindings(null);
    if (ws.current) {
      ws.current.close();
    }
  };

  const renderAnalyzeButtonContent = () => {
    switch (status) {
      case APP_STATUS.IDLE:
        return `Analyze ${files.length} File(s)`;
      case APP_STATUS.UPLOADING:
        return (
          <>
            <span className="spinner" />
            Uploading...
          </>
        );
      case APP_STATUS.PROCESSING:
        return (
          <>
            <span className="spinner" />
            Analyzing...
          </>
        );
      case APP_STATUS.COMPLETE:
        return 'Analysis Complete';
      case APP_STATUS.ERROR:
        return 'Analysis Failed';
      default:
        return 'Analyze';
    }
  };

  return (
    <>
      <h1>Systems Engineering Command Center (SECC)</h1>
      <p>Upload system PDF files for conflict, gap, and inconsistency analysis.</p>

      {status === APP_STATUS.IDLE && (
        <FileUpload files={files} onFilesSelected={setFiles} />
      )}
      
      {status !== APP_STATUS.IDLE && (
        <AnalysisView logs={logs} />
      )}

      {status === APP_STATUS.COMPLETE && findings && (
        <ResultsDisplay findingsList={findings} />
      )}
      
      <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
        <button
          onClick={handleAnalyzeClick}
          disabled={status !== APP_STATUS.IDLE || files.length === 0}
          style={{ 
            minWidth: '200px', 
            display: 'inline-flex', 
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {renderAnalyzeButtonContent()}
        </button>

        {status !== APP_STATUS.IDLE && (
          <button onClick={handleReset}>
            Start Over
          </button>
        )}
      </div>
    </>
  );
}

export default App;