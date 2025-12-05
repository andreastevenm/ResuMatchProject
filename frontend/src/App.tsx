import React, { useState } from "react";

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [expectedSkills, setExpectedSkills] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFiles(Array.from(e.target.files));
  };

  const analyzeResume = async () => {
    if (files.length === 0) {
      alert("Upload at least one PDF!");
      return;
    }

    if (!expectedSkills.trim()) {
      alert("Please enter expected skills first!");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    // always send lowercase
    formData.append("expected_skills", expectedSkills.toLowerCase());

    try {
      const res = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        body: formData,
      });
      const json = await res.json();
      setResult(json);
    } catch (err) {
      console.error(err);
      alert("Backend error.");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-white px-12 py-5 font-sans">
      {/* Logo */}
      <div className="text-2xl font-semibold tracking-tight mb-1">
        Resumatch
      </div>

      {/* Header */}
      <header className="text-center mt-2 mb-6">
        <h1 className="text-4xl font-bold tracking-tight text-gray-900">
          Resumatch Analysis
        </h1>
        <p className="text-gray-500 mt-2 text-s">
          Go ahead and upload your resume and get insightful feedback made by{" "}
          <span className="font-bold">Andreas, Danieka, and Ray!</span>
        </p>
      </header>

      {/* MAIN CARD */}
      <div className="mt-6 max-w-[1500px] mx-auto flex gap-8">
        {/* LEFT PURPLE CARD */}
        <div className="w-1/3 bg-purple-300/60 rounded-3xl p-10 shadow-md">
          <div className="max-w-[90%] mx-auto">
            <h3 className="font-semibold text-2xl mb-6">Requirements</h3>

            <ol className="space-y-8 text-base leading-relaxed">
              <li>
                <b className="text-lg">1. Upload Resume</b>
                <p className="text-gray-700 mt-1">
                  Drag and drop your resume or select from your device.
                </p>
              </li>

              <li>
                <b className="text-lg">2. Wait for Progress</b>
                <p className="text-gray-700 mt-1">
                  Our AI will analyze your resume.
                </p>
              </li>

              <li>
                <b className="text-lg">3. Review Insights</b>
                <p className="text-gray-700 mt-1">
                  Get a detailed report of your resume!
                </p>
              </li>
            </ol>
          </div>
        </div>

        {/* RIGHT WHITE CARD */}
        <div className="w-2/3 bg-gray-100 rounded-3xl p-12 shadow-md">
          <div className="bg-white p-10 rounded-3xl border shadow-sm">
            {/* Expected Skills Input */}
            <h3 className="text-2xl font-semibold text-center mb-5">
              Expected Skills
            </h3>

            <input
              type="text"
              placeholder="e.g. Java, React, Spring Boot"
              value={expectedSkills}
              onChange={
                (e) => setExpectedSkills(e.target.value.toLowerCase()) // ALWAYS lowercase
              }
              className="w-full p-3 border rounded-xl mb-6 text-lg"
            />

            {/* Upload */}
            <div className="text-center">
              <label className="inline-block bg-purple-500 hover:bg-purple-600 transition text-white px-7 py-3 rounded-xl cursor-pointer text-lg">
                Browse Files
                <input
                  type="file"
                  accept="application/pdf"
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>

            {/* Selected Files */}
            {files.length > 0 && (
              <ul className="list-disc ml-6 mt-6 text-purple-700 text-lg">
                {files.map((f, i) => (
                  <li key={i}>{f.name}</li>
                ))}
              </ul>
            )}

            {/* Analyze Button */}
            <div className="mt-10 text-center">
              <button
                onClick={analyzeResume}
                disabled={!expectedSkills.trim()}
                className={`px-10 py-4 rounded-2xl text-lg font-semibold shadow-md text-white
                ${
                  expectedSkills.trim()
                    ? "bg-purple-600 hover:bg-purple-700 transition"
                    : "bg-gray-400 cursor-not-allowed"
                }`}
              >
                {loading ? "Analyzing..." : "ANALYZE"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* RESULTS */}
      {result && (
        <div className="mt-16 max-w-4xl mx-auto space-y-6 mb-10">
          {result.resumes?.map((res: any, idx: number) => (
            <div
              key={idx}
              className="bg-white p-8 rounded-3xl shadow-lg border"
            >
              <div className="flex justify-between">
                <div>
                  <h3 className="font-bold text-xl">{res.filename}</h3>
                  <p className="text-gray-500 text-sm">{res.message}</p>
                </div>

                <div className="text-right">
                  <div className="text-sm text-gray-500">Score</div>
                  <div
                    className={`text-3xl font-bold ${
                      res.matching_score >= 55
                        ? "text-green-600"
                        : "text-red-600"
                    }`}
                  >
                    {res.matching_score}%
                  </div>
                  <div
                    className={`text-lg font-semibold ${
                      res.recommendation === "Recommended"
                        ? "text-green-600"
                        : "text-red-600"
                    }`}
                  >
                    {res.recommendation}
                  </div>
                </div>
              </div>

              {!res.error && (
                <>
                  <p className="mt-5 text-lg">
                    <b>Email:</b> {res.email || "Not found"}
                  </p>
                  <p className="text-lg">
                    <b>Phone:</b> {res.phone || "Not found"}
                  </p>

                  <p className="mt-6 font-semibold text-lg">Skills Found:</p>
                  <ul className="list-disc ml-6 text-base">
                    {res.skills_found?.map((s: string, i: number) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>

                  <p className="mt-6 font-semibold text-lg">Educatio:</p>
                  <ul className="list-disc ml-6 text-base">
                    {res.education_found?.map((e: string, i: number) => (
                      <li key={i}>{e}</li>
                    ))}
                  </ul>

                  <p className="mt-6 text-sm">
                    <b>Preview:</b>{" "}
                    <span className="text-gray-700">
                      {res.text_preview?.slice(0, 400)}
                    </span>
                  </p>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
