import React, { useState } from "react";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const analyzeResume = async () => {
    if (!file) {
      alert("Please upload a PDF first!");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        body: formData,
      });

      const json = await res.json();
      setResult(json);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend.");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-white px-10 py-6 font-sans">
      <div className="text-2xl font-semibold tracking-tight mb-4">
        Resumatch
      </div>

      <header className="text-center mt-2">
        <h1 className="text-6xl font-bold tracking-tight text-gray-900">
          Resumatch Analysis
        </h1>
        <p className="text-gray-600 mt-4 text-lg">
          Go ahead and upload your resume and get insightful feedback.
        </p>
      </header>

      <div className="mt-10 flex justify-center">
        <div className="w-[95%] max-w-7xl grid grid-cols-3 gap-10 bg-gray-100 p-10 rounded-3xl">
          <aside className="col-span-1 bg-purple-300/70 rounded-3xl p-10 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-900 mb-8">
              Requirements
            </h2>

            <ul className="space-y-8 text-gray-900">
              <li>
                <p className="font-semibold text-lg">1. Upload Resume</p>
                <p className="text-gray-800 text-sm mt-1">
                  Drag and drop your resume file or select from your device.
                </p>
              </li>
              <li>
                <p className="font-semibold text-lg">2. Wait for Progress</p>
                <p className="text-gray-800 text-sm mt-1">
                  Our AI will analyze your resume.
                </p>
              </li>
              <li>
                <p className="font-semibold text-lg">3. Review Insights</p>
                <p className="text-gray-800 text-sm mt-1">
                  Get a detailed report!
                </p>
              </li>
            </ul>
          </aside>

          <section className="col-span-2 bg-white rounded-3xl p-12 shadow-md border-2 border-blue-300 flex flex-col items-center">
            <p className="text-xl font-semibold text-gray-800">
              Drag & Drop your PDF file here
            </p>
            <p className="text-gray-500 mt-2">or</p>

            <label className="mt-4 bg-purple-400 hover:bg-purple-500 text-white font-medium px-6 py-2 rounded-xl cursor-pointer transition">
              Browse Files
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {file && (
              <p className="mt-4 text-gray-700">
                Selected File:{" "}
                <span className="font-semibold text-purple-700">
                  {file.name}
                </span>
              </p>
            )}

            <button
              onClick={analyzeResume}
              className="mt-6 bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-xl font-semibold transition"
            >
              Analyze Resume
            </button>

            {loading && (
              <p className="mt-4 text-lg font-semibold text-gray-700 animate-pulse">
                Loading...
              </p>
            )}
          </section>
        </div>
      </div>

      {result && (
        <div className="max-w-4xl mx-auto mt-10 bg-white shadow-md p-8 rounded-2xl border">
          <h2 className="text-2xl font-bold mb-4 text-gray-900">
            🔍 Resume Analysis Result
          </h2>

          <p>
            <b>Email:</b> {result.email || "Not found"}
          </p>
          <p>
            <b>Phone:</b> {result.phone || "Not found"}
          </p>

          <p className="mt-4">
            <b>Skills Found:</b>
          </p>
          <ul className="list-disc pl-6 text-gray-800">
            {result.skills_found.map((skill: string, index: number) => (
              <li key={index}>{skill}</li>
            ))}
          </ul>

          <p className="mt-4">
            <b>Education (ORG):</b>
          </p>
          <ul className="list-disc pl-6">
            {result.education_found.map((edu: string, index: number) => (
              <li key={index}>{edu}</li>
            ))}
          </ul>

          <p className="mt-4 text-xl font-semibold">
            ⭐ Matching Score: {result.matching_score}%
          </p>

          <p className="mt-6">
            <b>Text Preview:</b>
            <br />
            <span className="text-gray-700">{result.text_preview}</span>
          </p>
        </div>
      )}
    </div>
  );
}
