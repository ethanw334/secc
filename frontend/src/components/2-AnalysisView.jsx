import React, { useRef, useEffect } from 'react';

function AnalysisView({ logs }) {
  const logEndRef = useRef(null);

  useEffect(() => {
    // Auto-scroll to the bottom of the log
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="card">
      <h3>Analysis Log</h3>
      <div className="log-container">
        {logs.map((log, index) => (
          <p key={index} className={log.type === 'error' ? 'error' : ''}>
            {`> ${log.message}`}
          </p>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}

export default AnalysisView;