export interface DataSample {
  id: string;
  image: HTMLImageElement;
  tensor?: any; // TFJS Tensor
  features?: number[]; // MobileNet embedding
  classId: string;
}

export interface ClassCategory {
  id: string;
  name: string;
  color: string;
  sampleCount: number;
}

export enum ModelType {
  LOGISTIC_REGRESSION = 'Logistic Regression',
  RANDOM_FOREST = 'Random Forest',
  CNN = 'CNN (Transfer Learning)',
  GEMINI = 'Gemini 2.5 Flash'
}

export interface ModelStatus {
  type: ModelType;
  isTraining: boolean;
  isTrained: boolean;
  accuracy: number;
  progress: number; // 0-100
  logs: string[];
}

export interface PredictionResult {
  modelType: ModelType;
  probabilities: { [classId: string]: number };
  predictedClassId: string;
  confidence: number;
  explanation?: string; // For Gemini
}

export type ConfusionMatrix = number[][]; // [actual][predicted]

export interface EvaluationMetrics {
  accuracy: number;
  confusionMatrix: ConfusionMatrix;
}