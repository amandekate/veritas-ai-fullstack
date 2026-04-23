"use client";

import { useState } from "react";
import {
  Settings,
  UploadCloud,
  Image as ImageIcon,
  FileText,
} from "lucide-react";

export default function Home() {
  const [headline, setHeadline] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const [textWeight, setTextWeight] = useState(0.6);
  const imageWeight = 1 - textWeight;

  const handleSubmit = async () => {
    if (!headline || !file) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("headline", headline);
    formData.append("file", file);
    formData.append("w_text", textWeight.toString());
    formData.append("w_image", imageWeight.toString());

    try {
      const res = await fetch(
        "https://amandekate-veritas-api.hf.space/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-[#0b1220] via-[#0f172a] to-[#020617] text-white">
      {/* Sidebar */}
      <div className="w-64 backdrop-blur-xl bg-white/5 border-r border-white/10 p-6 space-y-6 flex-shrink-0">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Settings size={18} />
          Settings
        </h2>

        <div>
          <p className="text-sm text-gray-400 mb-2">Fusion Balance</p>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={textWeight}
            onChange={(e) => setTextWeight(parseFloat(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>

        <div className="backdrop-blur-lg bg-white/10 p-4 rounded-xl border border-white/10">
          <p className="font-semibold mb-2">Current Weights</p>

          <div className="flex justify-between items-center">
            <span className="flex items-center gap-2">
              <FileText size={16} />
              Text
            </span>
            <span>{textWeight.toFixed(2)}</span>
          </div>

          <div className="flex justify-between items-center mt-2">
            <span className="flex items-center gap-2">
              <ImageIcon size={16} />
              Image
            </span>
            <span>{imageWeight.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 p-6 md:p-10">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <h1 className="text-3xl md:text-5xl font-bold mb-2 tracking-tight">
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Veritas AI
            </span>{" "}
            Multimodal Fake News Detector
          </h1>

          <p className="text-gray-400 mb-8 max-w-2xl">
            Detect misinformation using advanced multimodal AI combining text and image analysis.
          </p>

          {/* Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* LEFT */}
            <div className="space-y-5">
              <h2 className="text-xl font-semibold">
                Enter News Details
              </h2>

              <textarea
                className="w-full p-4 bg-white/5 border border-white/10 rounded-xl resize-none min-h-[120px] focus:ring-2 focus:ring-blue-500 outline-none backdrop-blur-lg"
                placeholder="Enter headline..."
                value={headline}
                onChange={(e) => setHeadline(e.target.value)}
              />

              {/* Upload */}
              <div className="border border-dashed border-white/20 p-6 rounded-xl text-center hover:border-blue-400 transition bg-white/5 backdrop-blur-lg">
                <UploadCloud className="mx-auto mb-3 text-gray-400" />

                <p className="text-gray-300">Drag and drop image</p>
                <p className="text-sm text-gray-400 mb-3">
                  Max 20MB • JPG, PNG
                </p>

                <label className="cursor-pointer bg-white/10 px-4 py-2 rounded-lg inline-block hover:bg-white/20 transition">
                  Browse files
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) =>
                      setFile(e.target.files?.[0] || null)
                    }
                  />
                </label>

                {file && (
                  <>
                    <p className="text-sm text-gray-400 mt-3">
                      {file.name}
                    </p>

                    <img
                      src={URL.createObjectURL(file)}
                      className="mt-3 rounded-lg max-h-40 mx-auto object-cover shadow-lg"
                    />
                  </>
                )}
              </div>

              <button
                onClick={handleSubmit}
                disabled={loading}
                className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                  loading
                    ? "bg-gray-600"
                    : "bg-gradient-to-r from-red-500 to-pink-500 hover:scale-105 active:scale-95"
                }`}
              >
                {loading ? "Analyzing..." : "Verify Authenticity"}
              </button>
            </div>

            {/* RIGHT */}
            <div className="space-y-5">
              <h2 className="text-xl font-semibold">
                Analysis Report
              </h2>

              <div className="backdrop-blur-xl bg-white/5 border border-white/10 p-6 rounded-xl min-h-[220px] flex items-center justify-center">
                {loading && (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-10 h-10 border-4 border-gray-600 border-t-blue-400 rounded-full animate-spin"></div>
                    <p className="text-gray-400">Running AI analysis...</p>
                  </div>
                )}

                {!loading && !result && (
                  <p className="text-gray-400">Awaiting input...</p>
                )}

                {!loading && result && (
                  <div className="w-full space-y-5 animate-fade-in">
                    <div>
                      <p className="text-sm text-gray-400">Prediction</p>
                      <p
                        className={`text-4xl font-bold ${
                          result.label === "FAKE"
                            ? "text-red-400"
                            : "text-green-400"
                        }`}
                      >
                        {result.label}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-gray-400 mb-1">
                        Confidence
                      </p>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full transition-all"
                          style={{
                            width: `${result.confidence * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>

                    <div className="text-sm text-gray-300 space-y-1">
                      <p>Text Score: {result.text_score}</p>
                      <p>Image Score: {result.image_score}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer */}
          <p className="text-xs text-gray-500 mt-10 text-center">
            Built with FastAPI, TensorFlow, and Next.js
          </p>
        </div>
      </div>
    </div>
  );
}