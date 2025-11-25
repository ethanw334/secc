import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './components/1-FileUpload';
import AnalysisView from './components/2-AnalysisView';
import ResultsDisplay from './components/3-ResultsDisplay';

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
  const [selectedModel, setSelectedModel] = useState('qwen3:8b'); 
  const [localModels, setLocalModels] = useState([]);
  const [logs, setLogs] = useState([]);
  const [findings, setFindings] = useState(null);
  const [healthReport, setHealthReport] = useState(null);
  const ws = useRef(null);

  useEffect(() => {
    axios.get(`${API_URL}/models`)
      .then(response => {
        const models = response.data.models;
        setLocalModels(models);
        const hasQwen = models.find(m => m.name === 'qwen3:8b');
        if (hasQwen) {
            setSelectedModel('qwen3:8b');
        } else if (models.length > 0) {
            setSelectedModel(models[0].name);
        } else {
            setSelectedModel('gpt-5-mini');
        }
      })
      .catch(err => {
        console.error("Failed to fetch models", err);
        setSelectedModel('gpt-5-mini');
      });
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
      // HTTP Upload
      const uploadResponse = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      const { session_id } = uploadResponse.data;
      addLog('Upload complete. Connecting to analysis server...');
      setStatus(APP_STATUS.PROCESSING);

      // WebSocket Connection
      ws.current = new WebSocket(`${API_URL.replace('http', 'ws')}/ws/analysis`);

      ws.current.onopen = () => {
        addLog('Connection open. Starting analysis...');
        ws.current.send(JSON.stringify({ 
          session_id: session_id,
          model: selectedModel
        }));
      };

      ws.current.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'log') {
          addLog(msg.message);
        } else if (msg.type === 'result') {
          addLog('Analysis complete! Results received.');
          setFindings(msg.data.findings);
          setHealthReport(msg.data.health_report);
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
    setHealthReport(null);
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

      {status === APP_STATUS.COMPLETE && healthReport && (
        <div className={`card health-report-card ${healthReport.state_level}`}>
          <h3>Overall Health Report</h3>
          <div className="health-grid">
            <div className="health-score-container">
              <span className="health-score">{healthReport.score}</span>
              <span className="health-score-label">/ 100</span>
            </div>
            <div className="health-summary">
              <p className="health-state-message">{healthReport.state_message}</p>
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem', fontSize: '0.9em', color: '#888' }}>
                <span>
                Time: <strong>{healthReport.execution_time}</strong>
                </span>
                <span>
                Cost: <strong style={{ color: '#fff' }}>{healthReport.cost}</strong> <span style={{fontSize:'0.8em'}}>{healthReport.token_info}</span>
                </span>
              </div>
              
              <p>Found <strong>{healthReport.total_findings}</strong> total issues across {files.length} documents.</p>
              <div className="health-counts">
                <span><strong style={{color: '#ff4d4d'}}>Critical:</strong> {healthReport.critical_count}</span>
                <span><strong style={{color: '#ffa500'}}>High:</strong> {healthReport.high_count}</span>
                <span><strong style={{color: '#ffd700', colorScheme: 'light'}}>Medium:</strong> {healthReport.medium_count}</span>
                <span><strong style={{color: '#52c41a'}}>Low:</strong> {healthReport.low_count}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {status === APP_STATUS.COMPLETE && findings && (
        <ResultsDisplay findingsList={findings} />
      )}
      
      {/* Dynamic Model Selector */}
      <div style={{ marginTop: '1rem', marginBottom: '1rem' }}>
        <label style={{ marginRight: '10px', fontWeight: 'bold' }}>Select AI Model:</label>
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={status !== APP_STATUS.IDLE}
          style={{ padding: '0.5rem', borderRadius: '4px', backgroundColor: '#333', color: '#fff', border: '1px solid #555' }}
        >
          {/* Cloud Options */}
          <optgroup label="Cloud (High Performance)">
            <option value="gpt-5-mini">OpenAI GPT-5 Mini</option>
          </optgroup>

          {/* Local Options */}
          <optgroup label="Local (Ollama)">
            {localModels.length === 0 && <option disabled>No local models found</option>}
            
            {localModels.map((modelObj) => (
              <option key={modelObj.name} value={modelObj.name}>
                {modelObj.name} ({modelObj.size_label})
              </option>
            ))}
          </optgroup>
        </select>
      </div>

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