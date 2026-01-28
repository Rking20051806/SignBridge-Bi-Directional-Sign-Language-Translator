import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Video, StopCircle, Play, Loader2, LayoutGrid, Scan, Image as ImageIcon, CheckCircle2, Fingerprint, AlertCircle, Maximize2, User, Wifi, Activity, Zap, Search, Aperture, Target, Settings2, Gauge, ShieldCheck, ShieldAlert, XCircle } from 'lucide-react';
import { analyzeSignLanguageFrame, detectSignWithBoundingBox } from '../services/geminiService';
import { Detection } from '../types';
import { ASL_ALPHABET } from '../constants';
import { HolisticLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Connection maps
const HAND_CONNECTIONS = [
  {start: 0, end: 1}, {start: 1, end: 2}, {start: 2, end: 3}, {start: 3, end: 4},
  {start: 0, end: 5}, {start: 5, end: 6}, {start: 6, end: 7}, {start: 7, end: 8},
  {start: 5, end: 9}, {start: 9, end: 10}, {start: 10, end: 11}, {start: 11, end: 12},
  {start: 9, end: 13}, {start: 13, end: 14}, {start: 14, end: 15}, {start: 15, end: 16},
  {start: 13, end: 17}, {start: 0, end: 17}, {start: 17, end: 18}, {start: 18, end: 19}, {start: 19, end: 20}
];

const POSE_CONNECTIONS = [
  {start: 11, end: 12}, {start: 11, end: 13}, {start: 13, end: 15}, 
  {start: 12, end: 14}, {start: 14, end: 16}
];

type InputMode = 'webcam' | 'ip-cam';

const SignToText: React.FC = () => {
  const [inputMode, setInputMode] = useState<InputMode>('webcam');
  const [ipCamUrl, setIpCamUrl] = useState<string>('http://192.168.1.5:4747/video');
  
  // Settings State
  const [stabilization, setStabilization] = useState(true);
  const [captureInterval, setCaptureInterval] = useState(1500); // Optimized default
  const [showSettings, setShowSettings] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null); 
  const cropCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // MediaPipe
  const holisticLandmarkerRef = useRef<HolisticLandmarker | null>(null);
  const requestRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  
  // Logic State
  const activeLandmarksRef = useRef<any>(null);
  const predictionBufferRef = useRef<string[]>([]); // For stabilization
  
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentSign, setCurrentSign] = useState<string>("...");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [matchedLetter, setMatchedLetter] = useState<string | null>(null);
  const [handPresent, setHandPresent] = useState(false);
  
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  useEffect(() => {
    const loadMediaPipe = async () => {
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );
            
            holisticLandmarkerRef.current = await HolisticLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`,
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
            });
            setIsModelLoading(false);
        } catch (err: any) {
            console.error(err);
            setError("Vision Module Failed to Load. Check internet connection.");
            setIsModelLoading(false);
        }
    };
    loadMediaPipe();
  }, []);

  // Manage Analysis Loop based on state
  useEffect(() => {
      if (isAnalyzing && isStreaming) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          intervalRef.current = window.setInterval(captureAndAnalyze, captureInterval);
      } else {
          if (intervalRef.current) clearInterval(intervalRef.current);
          intervalRef.current = null;
      }
      return () => {
          if (intervalRef.current) clearInterval(intervalRef.current);
      };
  }, [isAnalyzing, isStreaming, captureInterval, stabilization]); // Re-bind if settings change

  const startCamera = async () => {
    setError(null);
    if (!videoRef.current) return;

    try {
      if (inputMode === 'webcam') {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } 
        });
        videoRef.current.srcObject = stream;
      } else if (inputMode === 'ip-cam') {
        videoRef.current.crossOrigin = "anonymous";
        videoRef.current.src = ipCamUrl;
      }

      await videoRef.current.play();
      videoRef.current.addEventListener('loadeddata', predictLoop);
      setIsStreaming(true);
    } catch (err: any) {
      setError("Camera Access Denied. Please allow permission or check device.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current) {
        if (videoRef.current.srcObject) {
            (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        videoRef.current.pause();
    }
    setIsStreaming(false);
    setIsAnalyzing(false);
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    clearCanvas();
    setHandPresent(false);
    activeLandmarksRef.current = null;
    predictionBufferRef.current = [];
  };

  const predictLoop = async () => {
    if (!videoRef.current || !overlayCanvasRef.current) return;
    
    if (holisticLandmarkerRef.current) {
        let startTimeMs = performance.now();
        if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = videoRef.current.currentTime;
            try {
                const result = holisticLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
                
                const left = result.leftHandLandmarks?.[0];
                const right = result.rightHandLandmarks?.[0];
                const pose = result.poseLandmarks?.[0];

                const hasHands = !!(left || right);
                setHandPresent(hasHands);
                
                // Store landmarks for the cropper to use
                activeLandmarksRef.current = { left, right, pose };

                drawHolisticLandmarks(result);
            } catch (e) { /* ignore */ }
        }
    }

    if (!videoRef.current.paused && !videoRef.current.ended) {
        requestRef.current = requestAnimationFrame(predictLoop);
    }
  };

  const drawHolisticLandmarks = (result: any) => {
    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas || !videoRef.current) return;

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Pose
    if (result.poseLandmarks?.[0]) {
        drawConnectors(ctx, result.poseLandmarks[0], POSE_CONNECTIONS, { 
            color: "rgba(56, 189, 248, 0.6)", 
            lineWidth: 6 
        });
        drawLandmarks(ctx, result.poseLandmarks[0], { 
            color: "rgba(56, 189, 248, 1)", 
            radius: 6 
        });
    }

    // Draw Hands & Bounding Box
    [result.leftHandLandmarks?.[0], result.rightHandLandmarks?.[0]].forEach(hand => {
        if (hand) {
            drawConnectors(ctx, hand, HAND_CONNECTIONS, { 
                color: "rgba(52, 211, 153, 1)", 
                lineWidth: 8 
            });
            drawLandmarks(ctx, hand, { 
                color: "#ffffff", 
                radius: 9 
            });
            
            const { minX, minY, width, height } = getBoundingBox(hand, canvas.width, canvas.height);
            ctx.strokeStyle = "#34d399";
            ctx.lineWidth = 2;
            
            const lineLen = 20;
            ctx.beginPath(); ctx.moveTo(minX, minY + lineLen); ctx.lineTo(minX, minY); ctx.lineTo(minX + lineLen, minY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX + width - lineLen, minY); ctx.lineTo(minX + width, minY); ctx.lineTo(minX + width, minY + lineLen); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX, minY + height - lineLen); ctx.lineTo(minX, minY + height); ctx.lineTo(minX + lineLen, minY + height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX + width - lineLen, minY + height); ctx.lineTo(minX + width, minY + height); ctx.lineTo(minX + width, minY + height - lineLen); ctx.stroke();
        }
    });
  };

  const getBoundingBox = (landmarks: any[], width: number, height: number) => {
      const x = landmarks.map(l => l.x);
      const y = landmarks.map(l => l.y);
      const minX = Math.min(...x) * width;
      const maxX = Math.max(...x) * width;
      const minY = Math.min(...y) * height;
      const maxY = Math.max(...y) * height;
      
      const padding = 40;
      return {
          minX: Math.max(0, minX - padding),
          minY: Math.max(0, minY - padding),
          width: Math.min(width, (maxX - minX) + (padding * 2)),
          height: Math.min(height, (maxY - minY) + (padding * 2))
      };
  };

  const drawConnectors = (ctx: CanvasRenderingContext2D, landmarks: any[], connections: any[], style: any) => {
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.lineWidth;
      ctx.beginPath();
      for (const c of connections) {
          const f = landmarks[c.start];
          const t = landmarks[c.end];
          ctx.moveTo(f.x * ctx.canvas.width, f.y * ctx.canvas.height);
          ctx.lineTo(t.x * ctx.canvas.width, t.y * ctx.canvas.height);
      }
      ctx.stroke();
  };

  const drawLandmarks = (ctx: CanvasRenderingContext2D, landmarks: any[], style: any) => {
      ctx.fillStyle = style.color;
      for (const l of landmarks) {
          ctx.beginPath();
          ctx.arc(l.x * ctx.canvas.width, l.y * ctx.canvas.height, style.radius, 0, 2 * Math.PI);
          ctx.fill();
      }
  };

  const clearCanvas = () => {
    const ctx = overlayCanvasRef.current?.getContext('2d');
    ctx?.clearRect(0, 0, overlayCanvasRef.current?.width || 0, overlayCanvasRef.current?.height || 0);
  };

  // Analysis Loop
  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !cropCanvasRef.current || !isStreaming) return; 
    
    // Determine if we have a hand to crop
    const landmarks = activeLandmarksRef.current;
    let imageToAnalyze = "";

    if (landmarks && (landmarks.left || landmarks.right)) {
        const hand = landmarks.left || landmarks.right;
        const { minX, minY, width, height } = getBoundingBox(hand, videoRef.current.videoWidth, videoRef.current.videoHeight);
        
        cropCanvasRef.current.width = width;
        cropCanvasRef.current.height = height;
        const ctx = cropCanvasRef.current.getContext('2d');
        
        ctx?.drawImage(
            videoRef.current, 
            minX, minY, width, height,
            0, 0, width, height
        );
        imageToAnalyze = cropCanvasRef.current.toDataURL('image/jpeg', 0.8);
    } else {
        return; // Skip if no hands detected to save API calls
    }

    try {
      const result = await analyzeSignLanguageFrame(imageToAnalyze);
      if (!result || result === "...") return;

      const cleanText = result.trim().toUpperCase();
      
      if (cleanText.length === 1 && ASL_ALPHABET[cleanText.toLowerCase()]) {
          let finalSign = cleanText;

          // Stabilization Logic
          if (stabilization) {
              const buffer = predictionBufferRef.current;
              buffer.push(cleanText);
              if (buffer.length > 3) buffer.shift(); // Keep last 3 frames
              
              // Majority Vote
              const counts = buffer.reduce((acc: any, val) => { acc[val] = (acc[val] || 0) + 1; return acc; }, {});
              const maxCount = Math.max(...Object.values(counts) as number[]);
              const candidate = Object.keys(counts).find(key => counts[key] === maxCount);

              // Require at least 2 consistent frames
              if (maxCount < 2) return;
              finalSign = candidate!;
          }

          setCurrentSign(prev => {
              if (prev !== finalSign) {
                  setHistory(h => [finalSign, ...h].slice(0, 8));
                  return finalSign;
              }
              return prev;
          });
          setMatchedLetter(finalSign.toLowerCase());
          setError(null);
      }
    } catch (err: any) { 
        console.error(err);
        // Only set error if it persists, but for now allow silent failure to keep UI clean
    } 
  }, [isStreaming, stabilization]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 h-full">
      {/* Main Interface */}
      <div className="xl:col-span-8 flex flex-col gap-4">
        
        {/* Control Bar */}
        <div className="bg-slate-900/80 p-1.5 rounded-xl flex items-center justify-between border border-slate-700/50 backdrop-blur-md relative z-20">
          <div className="flex gap-1">
            <div className="px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider bg-slate-700 text-white shadow-inner">
              Live Feed
            </div>
          </div>
          
          <div className="flex items-center gap-3 px-2">
            {isStreaming && (
              <div className="flex items-center gap-2 text-[10px] font-mono text-emerald-400 mr-2">
                <Target className={`w-3 h-3 ${handPresent ? 'animate-ping' : 'opacity-20'}`} />
                {handPresent ? "LOCKED" : "SCAN"}
              </div>
            )}
            
            {/* Settings Toggle */}
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-lg transition-all ${showSettings ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`}
              title="Settings"
            >
              <Settings2 className="w-4 h-4" />
            </button>
          </div>

          {/* Settings Popover */}
          {showSettings && (
            <div className="absolute top-full right-0 mt-2 w-64 bg-slate-800 border border-slate-700 rounded-xl shadow-xl p-4 flex flex-col gap-4 animate-in slide-in-from-top-2 z-50">
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <Gauge className="w-3 h-3" /> Capture Speed
                </label>
                <div className="grid grid-cols-3 gap-1 bg-slate-900 p-1 rounded-lg">
                  <button onClick={() => setCaptureInterval(2500)} className={`text-[10px] font-bold py-1 rounded ${captureInterval === 2500 ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>SLOW</button>
                  <button onClick={() => setCaptureInterval(1500)} className={`text-[10px] font-bold py-1 rounded ${captureInterval === 1500 ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>NORM</button>
                  <button onClick={() => setCaptureInterval(800)} className={`text-[10px] font-bold py-1 rounded ${captureInterval === 800 ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>FAST</button>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  {stabilization ? <ShieldCheck className="w-3 h-3 text-emerald-400" /> : <ShieldAlert className="w-3 h-3 text-amber-400" />} 
                  Stabilization
                </label>
                <button 
                  onClick={() => setStabilization(!stabilization)}
                  className={`w-full py-2 rounded-lg text-xs font-bold border transition-all ${stabilization ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : 'bg-slate-900 border-slate-700 text-slate-500'}`}
                >
                  {stabilization ? "ENABLED (SMOOTH)" : "DISABLED (RAW)"}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 flex items-center justify-between animate-in fade-in slide-in-from-top-2">
            <div className="flex items-center gap-2 text-red-400 text-xs font-bold">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300" title="Close"><XCircle className="w-4 h-4" /></button>
          </div>
        )}

        {/* Video Stage */}
        <div className="relative rounded-xl overflow-hidden bg-black aspect-video shadow-2xl border border-slate-800 ring-1 ring-slate-700/50 group">
          <div className="relative w-full h-full flex items-center justify-center">
            <video ref={videoRef} className={`w-full h-full object-contain transform ${inputMode === 'webcam' ? 'scale-x-[-1]' : ''} ${isStreaming ? 'opacity-100' : 'opacity-30'}`} crossOrigin="anonymous" playsInline />
            <canvas ref={overlayCanvasRef} className={`absolute inset-0 w-full h-full pointer-events-none transform ${inputMode === 'webcam' ? 'scale-x-[-1]' : ''} object-contain`} />
            
            {/* Status Overlay */}
            <div className="absolute top-4 left-4 flex gap-2">
              <div className="bg-black/50 backdrop-blur px-2 py-1 rounded border border-white/10 text-[10px] text-white/70 font-mono">
                CAM_01: {isStreaming ? "ONLINE" : "OFFLINE"}
              </div>
              {isAnalyzing && (
                <div className="bg-red-500/20 backdrop-blur px-2 py-1 rounded border border-red-500/30 text-[10px] text-red-400 font-mono animate-pulse">
                  REC
                </div>
              )}
            </div>

            {/* Scanner Line Effect */}
            {isAnalyzing && (
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent h-[20%] animate-scan pointer-events-none z-10" />
            )}

            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <Aperture className="w-16 h-16 text-slate-700 mb-4 opacity-50" />
                <button onClick={startCamera} disabled={isModelLoading} className="px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-full font-bold shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2">
                  {isModelLoading ? <Loader2 className="animate-spin w-4 h-4"/> : <Zap className="w-4 h-4" />}
                  ACTIVATE SYSTEM
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Action Bar */}
        {isStreaming && (
          <div className="grid grid-cols-2 gap-4">
            <button 
              onClick={() => setIsAnalyzing(!isAnalyzing)} 
              className={`py-4 rounded-xl font-black uppercase tracking-widest transition-all flex items-center justify-center gap-3 shadow-lg ${isAnalyzing ? 'bg-amber-500/10 text-amber-500 border border-amber-500/50 hover:bg-amber-500/20' : 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-emerald-500/20'}`}
            >
              {isAnalyzing ? <StopCircle className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isAnalyzing ? "PAUSE INFERENCE" : "START TRANSLATION"}
            </button>
            
            <button onClick={stopCamera} className="py-4 bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 rounded-xl font-bold uppercase tracking-widest border border-slate-700">
              TERMINATE
            </button>
          </div>
        )}
        
        {/* Hidden Crop Canvas */}
        <canvas ref={cropCanvasRef} className="hidden" />
      </div>

      {/* Info / Result Sidebar */}
      <div className="xl:col-span-4 flex flex-col gap-4">
        
        {/* Main Result Display */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-1 shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-500 opacity-50 group-hover:opacity-100 transition-opacity"></div>
          
          <div className="bg-slate-900/50 p-6 rounded-lg min-h-[240px] flex flex-col items-center justify-center relative">
            <h3 className="absolute top-4 left-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
              Detected Symbol
            </h3>
            
            <div className="text-8xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-400 drop-shadow-2xl">
              {currentSign}
            </div>
            
            {matchedLetter && (
              <div className="absolute bottom-4 right-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                <span className="text-emerald-500 text-xs font-bold tracking-widest">CONFIRMED</span>
              </div>
            )}
          </div>
        </div>

        {/* Reference Comparison */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 flex-1 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <Search className="w-3 h-3" /> Database Match
          </h3>
          
          <div className="flex-1 bg-black/40 rounded-lg border-2 border-dashed border-slate-700/50 flex items-center justify-center overflow-hidden relative">
            {matchedLetter ? (
              <div className="relative w-full h-full p-4 flex items-center justify-center animate-in zoom-in duration-300">
                <img 
                  src={ASL_ALPHABET[matchedLetter]} 
                  className="max-w-full max-h-[160px] object-contain drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                  alt="Reference"
                />
                <div className="absolute bottom-2 left-2 bg-slate-900/80 px-2 py-1 rounded text-[10px] text-slate-300 font-mono">
                  REF_IMG_{matchedLetter.toUpperCase()}.PNG
                </div>
              </div>
            ) : (
              <div className="text-center opacity-20">
                <Fingerprint className="w-12 h-12 mx-auto mb-2" />
                <p className="text-[10px] font-mono">NO MATCH FOUND</p>
              </div>
            )}
          </div>
        </div>

        {/* Log */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[200px] overflow-hidden flex flex-col">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Translation Log</h3>
          <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar">
            {history.map((h, i) => (
              <div key={i} className="flex items-center gap-3 text-sm font-mono text-slate-300 border-b border-slate-800/50 pb-1">
                <span className="text-slate-600 text-[10px]">{new Date().toLocaleTimeString([], {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'})}</span>
                <span className="text-emerald-400 font-bold">»</span>
                {h}
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default SignToText;
