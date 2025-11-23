import { DataSample, ClassCategory, ModelType, EvaluationMetrics } from '../types';

// Access global TFJS variables loaded via CDN
const tf = (window as any).tf;
const mobilenet = (window as any).mobilenet;

let featureExtractor: any = null;
let tfModels: Record<string, any> = {};

// Singleton to load MobileNet
export const loadFeatureExtractor = async () => {
  if (!featureExtractor) {
    console.log('Loading MobileNet...');
    // MobileNet V2 1.0 outputs 1280 features
    featureExtractor = await mobilenet.load({ version: 2, alpha: 1.0 });
    console.log('MobileNet Loaded');
  }
  return featureExtractor;
};

// Extract features from an image element
export const extractFeatures = async (img: HTMLImageElement): Promise<number[]> => {
  await loadFeatureExtractor();
  // infer returns a tensor, for MobileNet V2 1.0 this is 1280 dimensions
  const activation = featureExtractor.infer(img, true); 
  const data = await activation.data();
  activation.dispose();
  return Array.from(data);
};

// Convert dataset to Tensors
const prepareTensors = (samples: DataSample[], classes: ClassCategory[]) => {
  const classIdToIndex: Record<string, number> = {};
  classes.forEach((c, i) => classIdToIndex[c.id] = i);

  const xsData: number[][] = [];
  const ysData: number[] = [];

  samples.forEach(s => {
    if (s.features && classIdToIndex[s.classId] !== undefined) {
      xsData.push(s.features);
      ysData.push(classIdToIndex[s.classId]);
    }
  });

  if (xsData.length === 0) return null;

  const xs = tf.tensor2d(xsData);
  const ys = tf.oneHot(tf.tensor1d(ysData, 'int32'), classes.length);

  return { xs, ys, classIdToIndex };
};

// Train Logistic Regression (Simple Dense Layer)
export const trainLogisticRegression = async (
  samples: DataSample[],
  classes: ClassCategory[],
  onEpochEnd: (logs: any) => void
) => {
  const data = prepareTensors(samples, classes);
  if (!data) throw new Error("No data provided");

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: classes.length,
    activation: 'softmax',
    inputShape: [1280] // Updated from 1024 to 1280 for MobileNet V2
  }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  await model.fit(data.xs, data.ys, {
    epochs: 20,
    batchSize: 16,
    callbacks: {
      onEpochEnd: (epoch: number, logs: any) => {
        onEpochEnd({ epoch, accuracy: logs.acc, loss: logs.loss });
      }
    }
  });

  tfModels[ModelType.LOGISTIC_REGRESSION] = model;
  data.xs.dispose();
  data.ys.dispose();
  return model;
};

// Train CNN (Transfer Learning Head)
// Technically a Dense network on top of CNN features (Standard Transfer Learning)
export const trainCNN = async (
  samples: DataSample[],
  classes: ClassCategory[],
  onEpochEnd: (logs: any) => void
) => {
  const data = prepareTensors(samples, classes);
  if (!data) throw new Error("No data provided");

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    inputShape: [1280] // Updated from 1024 to 1280 for MobileNet V2
  }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: classes.length,
    activation: 'softmax'
  }));

  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  await model.fit(data.xs, data.ys, {
    epochs: 40,
    batchSize: 8,
    callbacks: {
      onEpochEnd: (epoch: number, logs: any) => {
        onEpochEnd({ epoch, accuracy: logs.acc, loss: logs.loss });
      }
    }
  });

  tfModels[ModelType.CNN] = model;
  data.xs.dispose();
  data.ys.dispose();
  return model;
};

export const predictTFJS = async (
  modelType: ModelType,
  features: number[],
  classes: ClassCategory[]
): Promise<number[]> => {
  const model = tfModels[modelType];
  if (!model) return new Array(classes.length).fill(0);

  const input = tf.tensor2d([features]);
  const prediction = model.predict(input);
  const data = await prediction.data();
  
  input.dispose();
  prediction.dispose();
  
  return Array.from(data);
};

export const disposeModels = () => {
  Object.values(tfModels).forEach((m: any) => m.dispose());
  tfModels = {};
};