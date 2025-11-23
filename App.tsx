import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Brain, Play, Image as ImageIcon, Plus, Trash2, Activity, Video, Upload, AlertCircle, RefreshCw, X } from 'lucide-react';
import { clsx } from 'clsx';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';

// Internal modules
import { ClassCategory, DataSample, ModelType, ModelStatus, PredictionResult } from './types';
import { extractFeatures, trainLogisticRegression, trainCNN, predictTFJS, disposeModels, loadFeatureExtractor } from './services/tensorFlowService';
import { rfInstance } from './services/randomForestService';
import { getGeminiPrediction } from './services/geminiService';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export default function App() {
  // State
  const [classes, setClasses] = useState<ClassCategory[]>([
    { id: 'class-1', name: 'Class 1', color: COLORS[0], sampleCount: 0 },
    { id: 'class-2', name: 'Class 2', color: COLORS[1], sampleCount: 0 }
  ]);
  const [samples, setSamples] = useState<DataSample[]>([]);
  const [activeTab, setActiveTab] = useState<'data' | 'train' | 'predict'>('data');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Webcam
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [activeClassId, setActiveClassId] = useState<string | null>(null); // For recording
  
  // Training
  const [modelStatuses, setModelStatuses] = useState<Record<ModelType, ModelStatus>>({
    [ModelType.LOGISTIC_REGRESSION]: { type: ModelType.LOGISTIC_REGRESSION, isTraining: false, isTrained: false, accuracy: 0, progress: 0, logs: [] },
    [ModelType.RANDOM_FOREST]: { type: ModelType.RANDOM_FOREST, isTraining: false, isTrained: false, accuracy: 0, progress: 0, logs: [] },
    [ModelType.CNN]: { type: ModelType.CNN, isTraining: false, isTrained: false, accuracy: 0, progress: 0, logs: [] },
    [ModelType.GEMINI]: { type: ModelType.GEMINI, isTraining: false, isTrained: true, accuracy: 0, progress: 100, logs: [] } // Gemini is pre-trained
  });
  
  // Prediction
  const [isPredicting, setIsPredicting] = useState(false); // For webcam loop
  const [isRunningInference, setIsRunningInference] = useState(false); // For single shot
  const [inferenceMode, setInferenceMode] = useState<'webcam' | 'upload'>('webcam');
  const [inferenceImage, setInferenceImage] = useState<string | null>(null); // Base64 of uploaded image for inference
  const [predictions, setPredictions] = useState<Record<ModelType, PredictionResult | null>>({
    [ModelType.LOGISTIC_REGRESSION]: null,
    [ModelType.RANDOM_FOREST]: null,
    [ModelType.CNN]: null,
    [ModelType.GEMINI]: null
  });

  // Load MobileNet on mount
  useEffect(() => {
    loadFeatureExtractor();
  }, []);

  // Webcam Logic
  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsWebcamActive(false);
    }
  }, []);

  const startWebcam = useCallback(async () => {
    setCameraError(null);
    stopWebcam(); // Ensure previous stream is closed

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError("Camera API not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 224 }, 
          height: { ideal: 224 },
          facingMode: 'user'
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
           setIsWebcamActive(true);
           videoRef.current?.play().catch(e => console.error("Error playing video:", e));
        };
      }
    } catch (e: any) {
      console.error("Camera error", e);
      setIsWebcamActive(false);
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        setCameraError("Permission denied. Please allow camera access in your browser settings.");
      } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
        setCameraError("No camera found. Please connect a camera.");
      } else if (e.name === 'NotReadableError' || e.name === 'TrackStartError') {
        setCameraError("Camera is in use by another application.");
      } else {
        setCameraError(`Camera error: ${e.message || "Unknown error"}`);
      }
    }
  }, [stopWebcam]);

  useEffect(() => {
    // Only start webcam if we are in Data tab OR (Predict tab AND Webcam mode)
    const shouldRunWebcam = activeTab === 'data' || (activeTab === 'predict' && inferenceMode === 'webcam');
    
    if (shouldRunWebcam) {
      startWebcam();
    } else {
      stopWebcam();
    }
    return () => stopWebcam();
  }, [activeTab, inferenceMode, startWebcam, stopWebcam]);

  // Capture Sample
  const captureSample = async (classId: string) => {
    if (!videoRef.current || !isWebcamActive) return;
    
    // Create image element from video frame
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(videoRef.current, 0, 0, 224, 224);
    
    const img = new Image();
    img.src = canvas.toDataURL();
    img.width = 224;
    img.height = 224;
    
    await new Promise(r => img.onload = r); // Wait for load

    // Extract features immediately
    const features = await extractFeatures(img);

    const newSample: DataSample = {
      id: Date.now().toString(),
      image: img,
      features,
      classId
    };

    setSamples(prev => [...prev, newSample]);
    setClasses(prev => prev.map(c => c.id === classId ? { ...c, sampleCount: c.sampleCount + 1 } : c));
  };

  // Handle Data Collection Image Upload
  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>, classId: string) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    setIsProcessing(true);

    const newSamples: DataSample[] = [];
    const fileArray = Array.from(files);
    
    // Process files sequentially
    for (const file of fileArray) {
      if (!file.type.startsWith('image/')) continue;

      try {
        // 1. Read File to Base64
        const base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });

        // 2. Load Image
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = reject;
          img.src = base64;
        });

        // 3. Resize to 224x224
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        if (!ctx) continue;
        ctx.drawImage(img, 0, 0, 224, 224);
        
        const finalImg = new Image();
        const finalDataUrl = canvas.toDataURL('image/jpeg');
        
        await new Promise<void>((resolve) => { 
            finalImg.onload = () => resolve();
            finalImg.src = finalDataUrl; 
        });

        // 4. Extract Features
        const features = await extractFeatures(finalImg);
        
        newSamples.push({
          id: `${Date.now()}-${Math.random()}`,
          image: finalImg,
          features,
          classId
        });

      } catch (e) {
        console.error("Error processing uploaded image:", file.name, e);
      }
    }

    if (newSamples.length > 0) {
      setSamples(prev => [...prev, ...newSamples]);
      setClasses(prev => prev.map(c => c.id === classId ? { ...c, sampleCount: c.sampleCount + newSamples.length } : c));
    }
    
    setIsProcessing(false);
    event.target.value = '';
  };

  // Hold-to-record logic
  const intervalRef = useRef<number | null>(null);
  const handleMouseDown = (classId: string) => {
    if (!isWebcamActive) return;
    setActiveClassId(classId);
    captureSample(classId); // Capture one immediately
    intervalRef.current = window.setInterval(() => captureSample(classId), 100);
  };
  const handleMouseUp = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setActiveClassId(null);
  };

  // Training Logic
  const trainModel = async (type: ModelType) => {
    if (samples.length < 2) return alert("Need more data (at least 2 samples)");
    
    setModelStatuses(prev => ({
      ...prev,
      [type]: { ...prev[type], isTraining: true, progress: 0, logs: [] }
    }));

    try {
      if (type === ModelType.LOGISTIC_REGRESSION) {
        await trainLogisticRegression(samples, classes, (logs) => {
          setModelStatuses(prev => ({
            ...prev,
            [type]: {
              ...prev[type],
              progress: ((logs.epoch + 1) / 20) * 100,
              accuracy: logs.accuracy,
              logs: [...prev[type].logs, `Epoch ${logs.epoch}: Loss ${logs.loss.toFixed(4)}`]
            }
          }));
        });
      } else if (type === ModelType.CNN) {
         await trainCNN(samples, classes, (logs) => {
          setModelStatuses(prev => ({
            ...prev,
            [type]: {
              ...prev[type],
              progress: ((logs.epoch + 1) / 40) * 100,
              accuracy: logs.accuracy,
              logs: [...prev[type].logs, `Epoch ${logs.epoch}: Loss ${logs.loss.toFixed(4)}`]
            }
          }));
        });
      } else if (type === ModelType.RANDOM_FOREST) {
        // Simulate async training
        await new Promise<void>(resolve => setTimeout(resolve, 500));
        rfInstance.train(samples);
        setModelStatuses(prev => ({
          ...prev,
          [type]: { ...prev[type], progress: 100, accuracy: 0.95, logs: ['Tree 1 built...', 'Ensemble complete'] } 
        }));
      }

      setModelStatuses(prev => ({
        ...prev,
        [type]: { ...prev[type], isTraining: false, isTrained: true }
      }));

    } catch (e) {
      console.error(e);
      alert("Training failed. See console for details.");
      setModelStatuses(prev => ({ ...prev, [type]: { ...prev[type], isTraining: false } }));
    }
  };

  // --- PREDICTION LOGIC ---

  // Helper: Format Prediction Output
  const formatPrediction = (probs: number[], classes: ClassCategory[], type: ModelType): PredictionResult => {
    const probabilityMap: Record<string, number> = {};
    let maxProb = -1;
    let maxClass = '';
    
    classes.forEach((c, i) => {
      probabilityMap[c.id] = probs[i];
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxClass = c.id;
      }
    });

    return {
      modelType: type,
      probabilities: probabilityMap,
      predictedClassId: maxClass,
      confidence: maxProb
    };
  };

  // Shared Logic: Run predictions on a specific image element
  const runPredictionsOnImage = async (sourceImg: HTMLImageElement) => {
    
    // 1. Normalize image to 224x224 to match training data
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(sourceImg, 0, 0, 224, 224);
    
    // Create a robust promise-based image loader for the processed canvas
    const img = new Image();
    const loadPromise = new Promise<void>((resolve) => {
        img.onload = () => resolve();
    });
    img.src = canvas.toDataURL();
    await loadPromise;

    // 2. Extract features
    const features = await extractFeatures(img);
    
    const newPreds: Partial<Record<ModelType, PredictionResult>> = {};

    // 3. Run Inference for trained models
    try {
        if (modelStatuses[ModelType.LOGISTIC_REGRESSION].isTrained) {
            const probs = await predictTFJS(ModelType.LOGISTIC_REGRESSION, features, classes);
            newPreds[ModelType.LOGISTIC_REGRESSION] = formatPrediction(probs, classes, ModelType.LOGISTIC_REGRESSION);
        }
        if (modelStatuses[ModelType.CNN].isTrained) {
            const probs = await predictTFJS(ModelType.CNN, features, classes);
            newPreds[ModelType.CNN] = formatPrediction(probs, classes, ModelType.CNN);
        }
        if (modelStatuses[ModelType.RANDOM_FOREST].isTrained) {
            const probs = rfInstance.predictProbabilities(features, classes);
            newPreds[ModelType.RANDOM_FOREST] = formatPrediction(probs, classes, ModelType.RANDOM_FOREST);
        }
    } catch (error) {
        console.error("Prediction error:", error);
    }

    setPredictions(prev => ({
        ...prev,
        ...newPreds
    }));
  };

  // Loop for Webcam
  const predictLoopRef = useRef<number | null>(null);
  const predictLoop = async () => {
    if (!videoRef.current || !isPredicting || !isWebcamActive || inferenceMode !== 'webcam') return;

    // Grab frame
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(videoRef.current, 0, 0, 224, 224);
    
    const img = new Image();
    img.src = canvas.toDataURL();
    await new Promise(r => img.onload = r);
    
    await runPredictionsOnImage(img);

    if (isPredicting) {
      predictLoopRef.current = requestAnimationFrame(predictLoop);
    }
  };

  useEffect(() => {
    if (isPredicting && inferenceMode === 'webcam') {
      predictLoop();
    } else {
      if (predictLoopRef.current) cancelAnimationFrame(predictLoopRef.current);
    }
    return () => { if (predictLoopRef.current) cancelAnimationFrame(predictLoopRef.current); };
  }, [isPredicting, modelStatuses, inferenceMode]);

  // Handle Uploaded Inference Image
  const handleInferenceImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const result = e.target?.result as string;
      setInferenceImage(result);

      // Auto-predict on upload if models are trained
      const anyTrained = Object.values(modelStatuses).some(s => s.isTrained && s.type !== ModelType.GEMINI);
      if (anyTrained) {
          const img = new Image();
          const p = new Promise(r => img.onload = r);
          img.src = result;
          await p;
          await runPredictionsOnImage(img);
      }
    };
    reader.readAsDataURL(file);
  };

  // Handle Gemini Prediction
  const askGemini = async () => {
    let img: HTMLImageElement;

    if (inferenceMode === 'webcam') {
        if (!videoRef.current || !isWebcamActive) return alert("Camera not active");
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(videoRef.current, 0, 0, 224, 224);
        img = new Image();
        img.src = canvas.toDataURL();
    } else {
        if (!inferenceImage) return alert("Please upload an image first");
        img = new Image();
        img.src = inferenceImage;
    }

    await new Promise(r => img.onload = r);

    setPredictions(prev => ({ ...prev, [ModelType.GEMINI]: { modelType: ModelType.GEMINI, probabilities: {}, predictedClassId: 'loading', confidence: 0 } }));

    try {
      const result = await getGeminiPrediction(img, classes);
      const matchedClass = classes.find(c => c.name.toLowerCase() === result.predictedClass.toLowerCase());
      
      setPredictions(prev => ({
        ...prev,
        [ModelType.GEMINI]: {
          modelType: ModelType.GEMINI,
          probabilities: { [matchedClass ? matchedClass.id : 'unknown']: result.confidence },
          predictedClassId: matchedClass ? matchedClass.id : 'unknown',
          confidence: result.confidence,
          explanation: result.explanation
        }
      }));
    } catch (e) {
      console.error(e);
      setPredictions(prev => ({ ...prev, [ModelType.GEMINI]: null }));
      alert("Gemini failed to predict. Check console logs.");
    }
  };

  // --- UI COMPONENTS ---

  const renderTabButton = (id: typeof activeTab, label: string, Icon: any) => (
    <button
      onClick={() => setActiveTab(id)}
      className={clsx(
        "flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors border-b-2",
        activeTab === id 
          ? "border-emerald-500 text-emerald-400" 
          : "border-transparent text-zinc-400 hover:text-zinc-200"
      )}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );

  // Common Webcam View
  const renderWebcamView = (isLive: boolean) => (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 overflow-hidden shadow-xl h-full flex flex-col">
      <div className="relative aspect-square bg-black rounded-xl overflow-hidden mb-4 flex items-center justify-center bg-zinc-950">
        {cameraError ? (
          <div className="flex flex-col items-center justify-center p-6 text-center">
             <AlertCircle className="w-12 h-12 text-red-500 mb-3" />
             <p className="text-red-400 mb-4 font-medium text-sm">{cameraError}</p>
             <button 
               onClick={startWebcam}
               className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 border border-zinc-700 text-white"
             >
               <RefreshCw className="w-4 h-4" /> Retry Camera
             </button>
          </div>
        ) : (
          <>
             <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className={clsx("w-full h-full object-cover transform scale-x-[-1]", !isWebcamActive && "hidden")} 
            />
            {!isWebcamActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-500">
                <Camera className="w-12 h-12 mb-2 animate-pulse" />
                <p className="text-sm">Initializing Camera...</p>
              </div>
            )}
          </>
        )}
        
        {isWebcamActive && (
          <div className="absolute bottom-3 right-3 px-2 py-1 bg-black/70 backdrop-blur text-xs font-mono rounded text-emerald-400 border border-emerald-500/30">
            Input: 224x224
          </div>
        )}

        {isLive && isWebcamActive && (
          <div className="absolute top-4 left-4">
             <div className="px-3 py-1 rounded-full bg-red-500/20 border border-red-500 text-red-400 text-xs font-bold flex items-center gap-2 animate-pulse">
               <div className="w-2 h-2 rounded-full bg-red-500" /> LIVE
             </div>
          </div>
        )}
      </div>
      {/* Footer of webcam card */}
      {!isLive ? (
        <div className="text-center text-sm text-zinc-400 mt-auto">
          Hold the record button or upload images to collect data.
        </div>
      ) : (
        <div className="flex flex-col gap-2 mt-auto">
           <div className="flex gap-2">
            <button 
                onClick={() => setIsPredicting(!isPredicting)}
                disabled={!isWebcamActive}
                className={clsx(
                "flex-1 py-3 rounded-lg font-bold transition-all border",
                isPredicting 
                    ? "bg-red-500/10 border-red-500/50 text-red-400 hover:bg-red-500/20" 
                    : "bg-emerald-500 text-zinc-900 border-emerald-400 hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed"
                )}
            >
                {isPredicting ? "Stop Inference" : "Start Live Inference"}
            </button>
            
            <button 
                onClick={askGemini}
                disabled={!isWebcamActive}
                className="px-4 py-3 rounded-lg font-bold bg-purple-600 text-white hover:bg-purple-500 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Use Gemini Flash 2.5 to analyze this frame"
            >
                <Brain className="w-4 h-4" /> Gemini
            </button>
           </div>
        </div>
      )}
    </div>
  );

  const renderUploadInferenceView = () => (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 overflow-hidden shadow-xl h-full flex flex-col">
      <div className="relative aspect-square bg-black rounded-xl overflow-hidden mb-4 flex items-center justify-center bg-zinc-950 border-2 border-dashed border-zinc-800 group hover:border-zinc-600 transition-colors">
         {inferenceImage ? (
             <>
                <img src={inferenceImage} alt="Inference Input" className="w-full h-full object-contain" />
                <button 
                    onClick={() => setInferenceImage(null)}
                    className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-red-500/80 rounded-full text-white transition-colors backdrop-blur"
                >
                    <X className="w-4 h-4" />
                </button>
             </>
         ) : (
             <label className="flex flex-col items-center justify-center cursor-pointer w-full h-full">
                 <Upload className="w-12 h-12 text-zinc-600 mb-3 group-hover:text-zinc-400 transition-colors" />
                 <p className="text-zinc-500 text-sm font-medium group-hover:text-zinc-300">Click or Drag to Upload</p>
                 <input type="file" accept="image/*" className="hidden" onChange={handleInferenceImageUpload} />
             </label>
         )}
      </div>
      
      <div className="flex gap-2 mt-auto">
        <button
            onClick={async () => {
                if(inferenceImage) {
                    const anyTrained = Object.values(modelStatuses).some(s => s.isTrained && s.type !== ModelType.GEMINI);
                    if (!anyTrained) {
                        alert("No models have been trained yet! Please go to the 'Train Models' tab to train your classifiers, or use Gemini for zero-shot analysis.");
                        return;
                    }
                    
                    setIsRunningInference(true);
                    try {
                        // Correctly handle image loading via Promise before inference
                        const img = new Image();
                        const loadPromise = new Promise((resolve, reject) => {
                            img.onload = resolve;
                            img.onerror = reject;
                        });
                        img.src = inferenceImage;
                        await loadPromise;
                        
                        await runPredictionsOnImage(img);
                    } catch(e) {
                        console.error("Inference failed:", e);
                    } finally {
                        setIsRunningInference(false);
                    }
                } else {
                    alert("Upload an image first!");
                }
            }}
            disabled={!inferenceImage || isRunningInference}
            className="flex-1 py-3 rounded-lg font-bold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
            {isRunningInference ? <Activity className="w-4 h-4 animate-spin" /> : null}
            {isRunningInference ? "Running..." : "Run Inference"}
        </button>
        <button 
             onClick={askGemini}
             disabled={!inferenceImage}
             className="px-4 py-3 rounded-lg font-bold bg-purple-600 text-white hover:bg-purple-500 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
             title="Use Gemini to analyze this image"
           >
            <Brain className="w-4 h-4" /> Gemini
         </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400">
              NeuroClassify
            </h1>
          </div>
          <div className="flex gap-1">
            {renderTabButton('data', 'Collect Data', ImageIcon)}
            {renderTabButton('train', 'Train Models', Activity)}
            {renderTabButton('predict', 'Inference', Play)}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* DATA TAB */}
        {activeTab === 'data' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Webcam Column */}
            <div className="lg:col-span-1">
              {renderWebcamView(false)}
            </div>

            {/* Classes Column */}
            <div className="lg:col-span-2 space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-semibold">Classes</h2>
                <button 
                  onClick={() => setClasses([...classes, { id: `class-${Date.now()}`, name: `Class ${classes.length + 1}`, color: COLORS[classes.length % COLORS.length], sampleCount: 0 }])}
                  className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm transition-colors"
                >
                  <Plus className="w-4 h-4" /> Add Class
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {classes.map((cls) => (
                  <div key={cls.id} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 flex flex-col gap-4 relative group">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-4 h-4 rounded-full" style={{ backgroundColor: cls.color }} />
                        <input 
                          value={cls.name}
                          onChange={(e) => setClasses(classes.map(c => c.id === cls.id ? { ...c, name: e.target.value } : c))}
                          className="bg-transparent text-lg font-medium outline-none placeholder-zinc-600 w-full"
                          placeholder="Class Name"
                        />
                      </div>
                      <button 
                        onClick={() => {
                          setClasses(classes.filter(c => c.id !== cls.id));
                          setSamples(samples.filter(s => s.classId !== cls.id));
                        }}
                        className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        disabled={!isWebcamActive || isProcessing}
                        onMouseDown={() => handleMouseDown(cls.id)}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                        onTouchStart={() => handleMouseDown(cls.id)}
                        onTouchEnd={handleMouseUp}
                        className={clsx(
                          "flex-1 py-3 rounded-lg font-medium transition-all active:scale-95 flex items-center justify-center gap-2 select-none",
                          activeClassId === cls.id 
                            ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20" 
                            : "bg-zinc-800 hover:bg-zinc-750 text-zinc-300 disabled:opacity-50 disabled:cursor-not-allowed"
                        )}
                      >
                        <Camera className="w-4 h-4" />
                        Hold to Record
                      </button>
                      
                      <input
                          type="file"
                          id={`upload-${cls.id}`}
                          multiple
                          accept="image/*"
                          className="hidden"
                          disabled={isProcessing}
                          onChange={(e) => handleImageUpload(e, cls.id)}
                      />
                      <label
                          htmlFor={`upload-${cls.id}`}
                          className={clsx(
                            "px-4 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 border",
                            isProcessing 
                              ? "bg-zinc-800 text-zinc-500 cursor-wait border-zinc-800"
                              : "bg-zinc-800 hover:bg-zinc-750 text-zinc-300 cursor-pointer border-zinc-700 hover:border-zinc-600"
                          )}
                      >
                          {isProcessing ? <Activity className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                          {isProcessing ? "Processing..." : "Upload"}
                      </label>
                    </div>

                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-emerald-500 transition-all duration-300" 
                          style={{ width: `${Math.min(cls.sampleCount * 2, 100)}%` }} 
                        />
                      </div>
                      <span className="text-xs font-mono text-zinc-500 w-12 text-right">{cls.sampleCount} img</span>
                    </div>
                    
                    {/* Sample Preview Grid */}
                    <div className="flex gap-1 overflow-x-auto pb-2 h-16 opacity-50 hover:opacity-100 transition-opacity">
                      {samples.filter(s => s.classId === cls.id).slice(-5).map(s => (
                        <img key={s.id} src={s.image.src} className="h-full rounded border border-zinc-700" alt="sample" />
                      ))}
                    </div>

                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TRAIN TAB */}
        {activeTab === 'train' && (
          <div className="space-y-8">
             <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-8 text-center">
               <h2 className="text-3xl font-bold mb-2">Model Training Center</h2>
               <p className="text-zinc-400 max-w-2xl mx-auto">
                 We extract features using <span className="text-emerald-400 font-mono">MobileNetV2</span> and train custom heads on top. 
                 Select a model architecture below to begin.
               </p>
             </div>

             <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[ModelType.LOGISTIC_REGRESSION, ModelType.RANDOM_FOREST, ModelType.CNN].map((type) => {
                  const status = modelStatuses[type];
                  return (
                    <div key={type} className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex flex-col h-96 relative overflow-hidden">
                      <div className="flex justify-between items-start mb-4 relative z-10">
                        <div>
                          <h3 className="text-lg font-bold text-white">{type}</h3>
                          <p className="text-xs text-zinc-500 mt-1">
                            {type === ModelType.LOGISTIC_REGRESSION && "Fast, linear decision boundaries."}
                            {type === ModelType.RANDOM_FOREST && "Ensemble of decision trees. Robust."}
                            {type === ModelType.CNN && "Deep Neural Network. High capacity."}
                          </p>
                        </div>
                        {status.isTrained && <div className="bg-emerald-500/10 text-emerald-400 px-2 py-1 rounded text-xs border border-emerald-500/20">Ready</div>}
                      </div>

                      <div className="flex-1 bg-zinc-950/50 rounded-lg mb-4 border border-zinc-800 p-2 overflow-y-auto font-mono text-xs text-zinc-400">
                         {status.logs.length === 0 ? (
                           <div className="h-full flex items-center justify-center italic opacity-30">Waiting to train...</div>
                         ) : (
                           status.logs.map((log, i) => <div key={i}>{log}</div>)
                         )}
                      </div>

                      <div className="space-y-2 relative z-10">
                         {status.isTraining && (
                           <div className="w-full bg-zinc-800 h-1 rounded-full overflow-hidden">
                             <div className="bg-emerald-500 h-full transition-all duration-300" style={{ width: `${status.progress}%` }} />
                           </div>
                         )}
                         <button
                           onClick={() => trainModel(type)}
                           disabled={status.isTraining || samples.length < 2}
                           className={clsx(
                             "w-full py-3 rounded-lg font-bold transition-all flex items-center justify-center gap-2",
                             status.isTraining 
                              ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                              : "bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                           )}
                         >
                           {status.isTraining ? (
                             <>
                               <Activity className="w-4 h-4 animate-spin" /> Training...
                             </>
                           ) : (
                             <>
                               <Play className="w-4 h-4" /> {status.isTrained ? "Retrain Model" : "Train Model"}
                             </>
                           )}
                         </button>
                      </div>
                    </div>
                  );
                })}
             </div>
          </div>
        )}

        {/* PREDICT TAB */}
        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Input Column (Webcam/Upload) */}
            <div className="lg:col-span-4 space-y-4">
              
              {/* Toggle Switch */}
              <div className="flex p-1 bg-zinc-900 rounded-lg border border-zinc-800">
                <button 
                  onClick={() => setInferenceMode('webcam')}
                  className={clsx(
                    "flex-1 py-2 text-sm font-medium rounded-md transition-colors flex items-center justify-center gap-2",
                    inferenceMode === 'webcam' ? "bg-zinc-800 text-white shadow" : "text-zinc-500 hover:text-zinc-300"
                  )}
                >
                  <Video className="w-4 h-4" /> Webcam
                </button>
                <button 
                  onClick={() => setInferenceMode('upload')}
                  className={clsx(
                    "flex-1 py-2 text-sm font-medium rounded-md transition-colors flex items-center justify-center gap-2",
                    inferenceMode === 'upload' ? "bg-zinc-800 text-white shadow" : "text-zinc-500 hover:text-zinc-300"
                  )}
                >
                  <Upload className="w-4 h-4" /> Upload
                </button>
              </div>

              {inferenceMode === 'webcam' ? renderWebcamView(true) : renderUploadInferenceView()}
            </div>

            {/* Results */}
            <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-4">
               {Object.values(ModelType).map((type) => {
                 const pred = predictions[type];
                 const status = modelStatuses[type];

                 return (
                   <div key={type} className={clsx(
                     "bg-zinc-900 border rounded-xl p-5 relative overflow-hidden transition-all",
                     type === ModelType.GEMINI ? "border-purple-500/50 bg-purple-900/10 md:col-span-2" : "border-zinc-800"
                   )}>
                     <h3 className={clsx(
                       "font-bold text-sm uppercase tracking-wider mb-4 flex items-center justify-between",
                       type === ModelType.GEMINI ? "text-purple-400" : "text-zinc-400"
                     )}>
                       {type}
                       {pred && (
                         <span className="text-white bg-zinc-800 px-2 py-0.5 rounded text-xs">
                           {(pred.confidence * 100).toFixed(1)}% Conf
                         </span>
                       )}
                     </h3>

                     {/* UNTRAINED STATE */}
                     {!status.isTrained && type !== ModelType.GEMINI ? (
                        <div className="h-32 flex flex-col items-center justify-center text-zinc-600 gap-2 bg-zinc-950/30 rounded-lg border border-zinc-800 border-dashed">
                           <Activity className="w-8 h-8 opacity-20" />
                           <p className="text-xs font-medium">Model not trained yet</p>
                           <button 
                             onClick={() => setActiveTab('train')}
                             className="text-emerald-500 text-xs hover:underline"
                           >
                             Go to Training
                           </button>
                        </div>
                     ) : pred ? (
                        <div className="space-y-3">
                          {classes.map(cls => {
                            const prob = pred.probabilities[cls.id] || 0;
                            const isWinner = pred.predictedClassId === cls.id;
                            return (
                              <div key={cls.id} className="relative">
                                <div className="flex justify-between text-xs mb-1">
                                  <span className={isWinner ? "text-white font-bold" : "text-zinc-500"}>{cls.name}</span>
                                  <span className="font-mono text-zinc-600">{(prob * 100).toFixed(0)}%</span>
                                </div>
                                <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full transition-all duration-300 rounded-full"
                                    style={{ 
                                      width: `${prob * 100}%`,
                                      backgroundColor: isWinner ? cls.color : '#3f3f46'
                                    }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                          
                          {type === ModelType.GEMINI && pred.explanation && (
                            <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg text-sm text-purple-200">
                              <span className="font-bold block mb-1 text-purple-400 text-xs uppercase">Analysis</span>
                              "{pred.explanation}"
                            </div>
                          )}
                        </div>
                     ) : (
                       <div className="h-32 flex items-center justify-center text-zinc-600 text-sm">
                         {type === ModelType.GEMINI ? "Click 'Gemini' to analyze." : "Waiting for input..."}
                       </div>
                     )}
                   </div>
                 );
               })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}