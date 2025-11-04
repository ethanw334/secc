import React from 'react';
import ReactMarkdown from 'react-markdown'; // <-- 1. IMPORT THE LIBRARY

function ResultsDisplay({ findingsList }) {
  const { inconsistencies } = findingsList;

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'Critical': return '#ff4d4d';
      case 'High': return '#ffa500';
      case 'Medium': return '#ffd700';
      case 'Low': return '#52c41a';
      default: return '#888';
    }
  };

  return (
    <div className="card">
      <h2>Analysis Results</h2>
      <p>Found <strong>{inconsistencies.length}</strong> total inconsistencies.</p>
      
      <hr style={{ margin: '1.5rem 0', borderColor: '#333' }} />
      
      {inconsistencies.length === 0 && (
        <p>No inconsistencies were found between the provided documents.</p>
      )}

      {inconsistencies.map((finding) => (
        <div key={finding.FindingID} className="finding">
          <div className="finding-header">
            <span className="finding-id">{finding.FindingID}</span>
            <span 
              className="finding-severity"
              style={{ 
                backgroundColor: getSeverityColor(finding.SeverityLevel),
                color: '#000',
              }}
            >
              {finding.SeverityLevel}
            </span>
          </div>
          
          <div style={{ margin: '0.5rem 0' }}>
            <strong>Category:</strong> {finding.Category}
          </div>
          <div style={{ margin: '0.5rem 0' }}>
            <strong>Confidence:</strong> {finding.ConfidenceScore * 100}%
          </div>
          <div style={{ margin: '0.5rem 0' }}>
            <strong>Involved Artifacts:</strong> {finding.SourceArtifacts.join(', ')}
          </div>
          
          {/* --- 2. USE THE MARKDOWN COMPONENT --- */}
          <div className="finding-text">
            <ReactMarkdown>
              {finding.FindingText}
            </ReactMarkdown>
          </div>
        </div>
      ))}
    </div>
  );
}

export default ResultsDisplay;