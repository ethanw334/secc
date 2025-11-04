import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

function FileUpload({ files, onFilesSelected }) {
  const onDrop = useCallback((acceptedFiles) => {
    onFilesSelected(acceptedFiles);
  }, [onFilesSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
  });

  return (
    <div className="card">
      <div 
        {...getRootProps()} 
        style={dropzoneStyle(isDragActive)}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the PDFs here ...</p>
        ) : (
          <p>Drag 'n' drop PDF files here, or click to select files</p>
        )}
      </div>
      {files.length > 0 && (
        <aside style={{ marginTop: '1rem' }}>
          <h4>Selected Files:</h4>
          <ul>
            {files.map(file => (
              <li key={file.path}>
                {file.path} - {Math.round(file.size / 1024)} KB
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
}

const dropzoneStyle = (isDragActive) => ({
  border: `2px dashed ${isDragActive ? '#646cff' : '#555'}`,
  borderRadius: '8px',
  padding: '2rem',
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'border .2s ease-in-out',
  backgroundColor: isDragActive ? '#333' : 'transparent',
});

export default FileUpload;