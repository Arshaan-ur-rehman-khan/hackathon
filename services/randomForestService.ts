import { DataSample, ClassCategory } from '../types';

// A simplified Decision Tree node
interface TreeNode {
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: string; // Class ID if leaf
}

class SimpleDecisionTree {
  root: TreeNode | null = null;
  maxDepth: number;

  constructor(maxDepth: number = 5) {
    this.maxDepth = maxDepth;
  }

  train(samples: DataSample[]) {
    this.root = this.buildTree(samples, 0);
  }

  // Simplified CART algorithm: Randomly select features to split for "Random" Forest effect
  private buildTree(samples: DataSample[], depth: number): TreeNode {
    // Stop conditions
    const classCounts: Record<string, number> = {};
    let dominantClass = samples[0]?.classId;
    let maxCount = 0;

    samples.forEach(s => {
      classCounts[s.classId] = (classCounts[s.classId] || 0) + 1;
      if (classCounts[s.classId] > maxCount) {
        maxCount = classCounts[s.classId];
        dominantClass = s.classId;
      }
    });

    // Pure node or max depth reached
    if (Object.keys(classCounts).length === 1 || depth >= this.maxDepth || samples.length < 2) {
      return { value: dominantClass };
    }

    // Try to find a split
    let bestGain = -1;
    let bestFeature = -1;
    let bestThreshold = 0;
    
    const numFeatures = samples[0].features!.length;
    // Only check a random subset of features (Random Subspace Method)
    const featuresToCheck = 20; 

    for (let i = 0; i < featuresToCheck; i++) {
      const featureIdx = Math.floor(Math.random() * numFeatures);
      
      // Pick a random sample to determine threshold
      const sampleIdx = Math.floor(Math.random() * samples.length);
      const threshold = samples[sampleIdx].features![featureIdx];

      const left = samples.filter(s => s.features![featureIdx] <= threshold);
      const right = samples.filter(s => s.features![featureIdx] > threshold);

      if (left.length === 0 || right.length === 0) continue;

      const gain = this.calculateGiniGain(samples, left, right);
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = featureIdx;
        bestThreshold = threshold;
      }
    }

    if (bestGain <= 0) {
      return { value: dominantClass };
    }

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: this.buildTree(samples.filter(s => s.features![bestFeature] <= bestThreshold), depth + 1),
      right: this.buildTree(samples.filter(s => s.features![bestFeature] > bestThreshold), depth + 1)
    };
  }

  private calculateGiniGain(parent: DataSample[], left: DataSample[], right: DataSample[]): number {
    const gini = (group: DataSample[]) => {
      const counts: Record<string, number> = {};
      group.forEach(s => counts[s.classId] = (counts[s.classId] || 0) + 1);
      let impurity = 1;
      Object.values(counts).forEach(c => {
        const prob = c / group.length;
        impurity -= prob * prob;
      });
      return impurity;
    };

    const p = parent.length;
    return gini(parent) - ((left.length / p) * gini(left) + (right.length / p) * gini(right));
  }

  predict(features: number[]): string {
    let node = this.root;
    while (node && !node.value) {
      if (features[node.featureIndex!] <= node.threshold!) {
        node = node.left || null;
      } else {
        node = node.right || null;
      }
    }
    return node?.value || '';
  }
}

// The Forest
export class RandomForestClassifier {
  trees: SimpleDecisionTree[] = [];
  numTrees: number = 10;

  train(samples: DataSample[]) {
    this.trees = [];
    for (let i = 0; i < this.numTrees; i++) {
      // Bootstrap sampling
      const bootstrap = Array.from({ length: samples.length }, () => 
        samples[Math.floor(Math.random() * samples.length)]
      );
      const tree = new SimpleDecisionTree(5); // Depth 5
      tree.train(bootstrap);
      this.trees.push(tree);
    }
  }

  predictProbabilities(features: number[], classes: ClassCategory[]): number[] {
    const votes: Record<string, number> = {};
    classes.forEach(c => votes[c.id] = 0);

    this.trees.forEach(tree => {
      const prediction = tree.predict(features);
      if (prediction && votes[prediction] !== undefined) {
        votes[prediction]++;
      }
    });

    // Softmax-ish normalization
    const total = this.trees.length;
    return classes.map(c => votes[c.id] / total);
  }
}

export const rfInstance = new RandomForestClassifier();