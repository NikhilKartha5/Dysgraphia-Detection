"use client";
import { useState } from "react";

export default function PredictPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<{ prediction: string; explanation: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!selectedFile) {
      setError("Please select an image file.");
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("API error");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Failed to get prediction. Is the Flask API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto py-12 px-4">
      <h1 className="text-2xl font-bold mb-6">Upload Handwriting Sample</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" className="btn bg-blue-600 text-white" disabled={loading}>
          {loading ? "Predicting..." : "Get Prediction"}
        </button>
      </form>
      {error && <div className="text-red-500 mt-4">{error}</div>}
      {result && (
        <div className="mt-6 p-4 bg-gray-100 rounded">
          <div className="font-semibold">Prediction: {result.prediction}</div>
          <div className="mt-2">Explanation: {result.explanation}</div>
        </div>
      )}
    </div>
  );
} 