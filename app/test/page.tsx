"use client";

import React, { useState } from "react";

export default function TestPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    all_data: number[];
    explanation: string;
    predictions: string | number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Convert uploaded image to base64
  const toBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        if (typeof reader.result === "string") {
          const base64 = reader.result.split(",")[1];
          resolve(base64);
        } else {
          reject("Failed to convert image to base64");
        }
      };
      reader.onerror = (error) => reject(error);
    });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const base64Image = await toBase64(selectedFile);

      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: base64Image }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Server error");
      }

      const data = await response.json();

      // Flatten all_data
      const flatData = Array.isArray(data.all_data) && Array.isArray(data.all_data[0])
        ? data.all_data[0]
        : data.all_data;

      setResult({
        all_data: flatData,
        explanation: data.explanation,
        predictions: data.predictions,
      });
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const featureLabels = [
    "Atypical Margin Usage",
    "Letter Inversions",
    "Letter Transpositions",
    "Spelling Errors",
    "Poor Legibility",
    "Abandoned Words",
    "Letter Reversals",
    "Incorrect Capitalization",
    "Letter or Word Crowding",
  ];

  return (
    <main className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">
        Upload Handwriting Sample for Dysgraphia Analysis
      </h1>

      <form onSubmit={handleSubmit} className="mb-8 flex flex-col items-center gap-4">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
          className="border rounded px-3 py-2"
          required
        />
        <button
          type="submit"
          className="btn bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          disabled={!selectedFile || loading}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {error && (
        <div className="text-red-600 mb-6 text-center font-semibold">{error}</div>
      )}

      {result && (
        <section>
          <h2 className="text-xl font-semibold mb-4 text-center">Analysis Results</h2>

          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
            {result.all_data.map((val, idx) => (
              <li
                key={idx}
                className={`p-4 rounded border ${
                  val === 1 ? "bg-red-100 border-red-400" : "bg-green-100 border-green-400"
                }`}
              >
                <strong>{featureLabels[idx]}:</strong>{" "}
                {val === 1 ? "Issue Detected" : "No Issue"}
              </li>
            ))}
          </ul>

          <div className="whitespace-pre-line p-4 bg-gray-100 rounded border border-gray-300">
            <h3 className="font-semibold mb-2">Explanation:</h3>
            <p>{result.explanation}</p>
          </div>

          <div className="whitespace-pre-line p-4 bg-purple-100 rounded border border-purple-400 mt-6">
            <h3 className="font-semibold mb-2">Prediction:</h3>
            <p>{result.predictions}</p>
          </div>
        </section>
      )}
    </main>
  );
}
