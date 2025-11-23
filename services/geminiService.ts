import { GoogleGenAI, Type } from "@google/genai";
import { ClassCategory } from '../types';

export const getGeminiPrediction = async (
  image: HTMLImageElement,
  classes: ClassCategory[]
) => {
  if (!process.env.API_KEY) {
    throw new Error("API Key not found");
  }

  // Convert image to base64
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not create canvas context");
  
  ctx.drawImage(image, 0, 0);
  const base64Data = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const classNames = classes.map(c => c.name).join(", ");

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: {
      parts: [
        {
          inlineData: {
            mimeType: 'image/jpeg',
            data: base64Data
          }
        },
        {
          text: `Analyze this image and classify it into one of the following classes: [${classNames}]. 
                 Return the predicted class name and a brief explanation of why.`
        }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          predictedClass: { type: Type.STRING },
          explanation: { type: Type.STRING },
          confidence: { type: Type.NUMBER, description: "Confidence score between 0 and 1" }
        },
        required: ["predictedClass", "explanation", "confidence"]
      }
    }
  });

  const text = response.text;
  if (!text) throw new Error("No response from Gemini");
  
  return JSON.parse(text);
};