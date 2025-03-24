import React, { useState } from "react";
import axios from "axios";

function App() {
    const [url, setUrl] = useState("");
    const [summary, setSummary] = useState("");
    const [brailleSummary, setBrailleSummary] = useState("");
    const [keyEntity, setKeyEntity] = useState("");
    const [error, setError] = useState("");

    const handleSummarize = async () => {
        setError("");
        setSummary("");
        setBrailleSummary("");
        setKeyEntity("");

        if (!url) {
            setError("Please enter a valid article URL.");
            return;
        }

        try {
            const response = await axios.post("http://127.0.0.1:8000/summarize", { url });
            setSummary(response.data.summary);
            setBrailleSummary(response.data.braille_summary);
            setKeyEntity(response.data.key_entity);
        } catch (err) {
            setError("Failed to fetch summary. Please try again.");
        }
    };

    return (
        <div style={{ textAlign: "center", padding: "20px", fontFamily: "Arial, sans-serif" }}>
            <h1>Article Summarizer & Braille Converter</h1>
            <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Enter article URL"
                style={{ width: "80%", padding: "10px", marginBottom: "10px", fontSize: "16px" }}
            />
            <br />
            <button 
                onClick={handleSummarize} 
                style={{ padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
            >
                Summarize
            </button>

            {error && <p style={{ color: "red", marginTop: "20px" }}>{error}</p>}

            {summary && (
                <div 
                    style={{ 
                        marginTop: "20px", 
                        textAlign: "left", 
                        width: "80%", 
                        margin: "auto", 
                        padding: "20px", 
                        border: "1px solid #ddd", 
                        borderRadius: "10px", 
                        backgroundColor: "#f9f9f9" 
                    }}
                >
                    <h2>Summary</h2>
                    <p>{summary}</p>

                    <h2>Key Person/Topic</h2>
                    {keyEntity ? (
                        <p>
                            <a 
                                href={`https://news.google.com/search?q=${encodeURIComponent(keyEntity)}`} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                style={{ fontWeight: "bold", color: "#1a0dab", textDecoration: "underline", cursor: "pointer" }}
                            >
                                {keyEntity}
                            </a>
                        </p>
                    ) : (
                        <p>No key topic found.</p>
                    )}

                    <h2>Braille Summary</h2>
                    <p style={{ fontSize: "24px", fontFamily: "Arial" }}>{brailleSummary}</p>
                </div>
            )}
        </div>
    );
}

export default App;
