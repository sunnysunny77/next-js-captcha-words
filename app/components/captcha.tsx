"use client";
import * as tf from "@tensorflow/tfjs";
import {useEffect, useRef, useState, useCallback} from "react";
import {getLabels, getClassify} from "@/lib/captcha";
import Image from "next/image";
import Spinner from "@/images/spinner.gif";

const CANVAS_WIDTH = 280;
const CANVAS_HEIGHT = 140;

const INVERT = false;

const Captcha = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const drawingRef = useRef(false);

  const [label, setLabel] = useState(Spinner.src);
  const [message, setMessage] = useState(null);
  const [disabled, setDisabled] = useState(false);

  const imageLoader = ({src, width}) => {

    return `${src}?w=${width}`;
  };

  const setRandomLabels = useCallback( async () => {
    try {
      const res = await getLabels();
      setLabel(res);
    } catch (err) {
      console.error(err);
      setMessage("Error");
    }
  },[]);

  const clear = async () => {
    if (!ctxRef.current) return;
    if (INVERT) {
      ctxRef.current.fillStyle = "white";
      ctxRef.current.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    } else {
      ctxRef.current.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
  };

  const getCanvasCoords = (
    event: PointerEvent,
    canvas: HTMLCanvasElement
  ) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;

    ctxRef.current = canvas.getContext("2d");

    const ctx = ctxRef.current!;
    const onPointerDown = (event: PointerEvent) => {
      drawingRef.current = true;
      const { x, y } = getCanvasCoords(event, canvas);
      ctx.strokeStyle = INVERT ? "black" : "white";
      const minDim = Math.min(canvas.width, canvas.height);
      ctx.lineWidth = Math.max(1, Math.round(minDim / 18));
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(x, y);
      event.preventDefault();
    };

    const onPointerMove = (event: PointerEvent) => {
      if (!drawingRef.current) return;
      const { x, y } = getCanvasCoords(event, canvas);
      ctx.lineTo(x, y);
      ctx.stroke();
      event.preventDefault();
    };

    const onPointerUp = () => {
      drawingRef.current = false;
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    ["pointerup", "pointercancel", "pointerleave"].forEach(evt =>
      canvas.addEventListener(evt, onPointerUp)
    );

    if (INVERT) {
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    } else {
      ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
    setRandomLabels();
    setMessage("Draw the word");

    return () => {
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      ["pointerup", "pointercancel", "pointerleave"].forEach(evt =>
        canvas.removeEventListener(evt, onPointerUp)
      );
    };
  }, [setRandomLabels]);

  const handleClear = async () => {
    clear();
    setMessage("Draw a capital letter in the boxes");
  };

  const handleReset = async () => {
    await setRandomLabels();
    clear();
    setMessage("Draw a capital letter in the boxes");
  };

  const resizeCanvas = (canvas) => {
    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 112;
    resizedCanvas.height = 28;
    const resizedCtx = resizedCanvas.getContext("2d");

    resizedCtx.drawImage(
      canvas,
      0, 0, CANVAS_WIDTH, CANVAS_HEIGHT,
      0, 0, 112, 28
    );

    return resizedCanvas;
  };

  const invertCanvas = (ctx) => {
    const image = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    const invertedCanvas = document.createElement("canvas");
    invertedCanvas.width = CANVAS_WIDTH;
    invertedCanvas.height = CANVAS_HEIGHT;
    const invertedCtx = invertedCanvas.getContext("2d");

    const invertedData = ctx.createImageData(image.width, image.height);

    for (let i = 0; i < image.data.length; i += 4) {
      invertedData.data[i] = 255 - image.data[i];
      invertedData.data[i + 1] = 255 - image.data[i + 1];
      invertedData.data[i + 2] = 255 - image.data[i + 2];
      invertedData.data[i + 3] = image.data[i + 3];
    };

    invertedCtx.putImageData(invertedData, 0, 0);

    return invertedCanvas;
  };

  const handleSubmit = async () => {
    try {
      setDisabled(true);
      setMessage(null);

      const canvas = INVERT ? invertCanvas(ctxRef.current) : canvasRef.current;
      const resizedCanvas = resizeCanvas(canvas);
      const img = tf.browser.fromPixels(resizedCanvas, 1).toFloat().div(255.0);

      const tensorData = {
        data: Array.from(new Uint8Array(img.mul(255).dataSync())),
        shape: img.shape
      };

      const result = await getClassify(tensorData);
      img.dispose();

      setMessage(result.correct ? "Correct" : "Incorrect");
    } catch (err) {
      console.error(err);
      setMessage("Error during prediction");
    } finally {
      setDisabled(false);
    }
  };

  return (

    <div className="hr-container d-flex flex-column" id="container">

      <h1 className="text-center mb-4">Handwritten recognition</h1>

      <div id="canvas-wrapper" className="mb-5 d-flex">

        <canvas ref={canvasRef} className="rounded shadow quad"/>

        <div className="fill"></div>

      </div>

      <div className="d-flex flex-wrap justify-content-center mb-5">

        <button className="btn btn-success m-2 p-1 button" onClick={handleReset}>New</button>

        <button className="btn btn-success m-2 p-1 button" onClick={handleClear}>Clear</button>

      </div>

      <div className="output-container mb-4">

        <div className="label-grid d-flex justify-content-center align-items-center">

          <Image width="280" height="112" src={label} loader={imageLoader} unoptimized alt="spinner"/>

        </div>

      </div>

      <div className="text-center alert alert-success w-100 d-flex justify-content-center align-items-center p-0 mb-4" role="alert">

        {message ? message : <Image width="30" height="30" src={Spinner} loader={imageLoader} unoptimized alt="spinner"/>}

      </div>

      <div className="d-flex flex-wrap justify-content-center">

        <button className="btn button btn-success w-100 p-1" disabled={disabled} onClick={handleSubmit}>

          Send

        </button>

      </div>

    </div>

  );

};

export default Captcha;
