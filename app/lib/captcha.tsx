"use server";
import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import {createCanvas} from "canvas";

const labels: string[] = [
  "THAT", "WITH", "HAVE", "FROM", "THIS", "WILL", "THEY", "WERE", "BEEN", "MORE",
  "SAID", "THEM", "THAN", "INTO", "MADE", "ONLY", "WHEN", "THIS", "LAST", "SOME",
  "THEY", "TIME", "GOOD", "ALSO", "MOST", "EVEN", "LIFE", "WHAT", "OVER", "WEST",
  "USED", "SUCH", "MANY", "WORK", "FILM", "MUST", "MUCH", "LIKE", "YOUR", "MAKE",
  "NEED", "DOWN", "TAKE", "MISS", "YEAR", "WELL", "PART", "PLAY", "WEEK", "DAYS"
];
let currentLabel: string = "";

let model: tf.GraphModel | null = null;

const loadModel = async (): Promise<void> => {
  if (!model) {
    const modelPath = path.join(process.cwd(), "tfjs_model/model.json");
    model = await tf.loadGraphModel(`file://${modelPath}`);
    console.log("Model loaded");
  }
};

const drawLabel = (label) => {
  const width = 250;
  const height = 100;
  const fill = "white";
  const dotCount = 250;
  const lineStyle = "rgba(0,0,0,0.34)";
  const lineWidth = 0.5;
  const fontSize = 30;
  const font = `bold ${fontSize}px Sans`;
  const overlap = 0.05;

  const word = label;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < dotCount; i++) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    const alpha = Math.random() < 0.5 ? 0.3 : 0.6;
    const radius = Math.random() * 3 + 1;
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, radius, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.strokeStyle = lineStyle;
  ctx.lineWidth = lineWidth;
  for (let j = 0; j < 2; j++) {
    ctx.beginPath();
    ctx.moveTo(0, Math.random() * canvas.height);
    for (let x = 0; x < canvas.width; x += 5) {
      ctx.lineTo(
        x,
        (canvas.height / 2) + Math.sin(x / 5 + Math.random() * 2) * 12 + (Math.random() * 20 - 10)
      );
    }
    ctx.stroke();
  }

  ctx.font = font;
  let totalWidth = 0;
  for (const char of word) {
    totalWidth += ctx.measureText(char).width * 0.8;
  }
  let x = (canvas.width - totalWidth) / 2;
  for (const char of word) {
    const angle = (Math.random() - 0.5) * 0.6;
    const offsetY = (Math.random() - 0.5) * 18;
    const min = 50;
    const max = 150;
    const r = Math.floor(Math.random() * (max - min) + min);
    const g = Math.floor(Math.random() * (max - min) + min);
    const b = Math.floor(Math.random() * (max - min) + min);
    const color = `rgba(${r},${g},${b},1)`;
    ctx.save();
    ctx.fillStyle = color;
    ctx.translate(x, canvas.height / 2 + offsetY);
    ctx.rotate(angle);
    ctx.fillText(char, 0, 0);
    ctx.restore();

    const overlapCalc = -ctx.measureText(char).width * overlap;
    x += ctx.measureText(char).width * 0.8 + overlapCalc;
  }

  return canvas.toDataURL();
};

export const getLabels = async () => {
  currentLabel = labels[Math.floor(Math.random() * labels.length)];

  return drawLabel(currentLabel);
};

const processImageNode = async (data, shape) => {
  const tensor = tf.tensor(data, shape, "float32").div(255.0);
  const input = tensor.resizeBilinear([28, 112]).expandDims(0);
  const prediction = model.predict(input) as tf.Tensor;
  const maxIndex = prediction.argMax(-1).dataSync()[0];

  tensor.dispose();
  input.dispose();
  prediction.dispose();

  return maxIndex;
};

export const getClassify = async (tensorData) => {
  if (!model) await loadModel();

  if (!currentLabel) {
    throw new Error("Error: No label available");
  }

  const predIndex = await processImageNode(tensorData.data, tensorData.shape);

  return {
    correct: currentLabel === labels[predIndex],
  };
};
