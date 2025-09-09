"use client";
import * as tf from "@tensorflow/tfjs";
import {useEffect, useRef, useState, useCallback} from "react";
import {getLabels, getClassify} from "@/lib/captcha";
import Image from "next/image";
import Spinner from "@/images/spinner.gif";

const CANVAS_WIDTH = 250;
const CANVAS_HEIGHT = 125;

const INVERT = false;

const Captcha = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const drawingRef = useRef(false);

  const [label, setLabel] = useState(Spinner.src);
  const [message, setMessage] = useState("Loading");
  const [disabled, setDisabled] = useState(false);

  const setRandomLabels = useCallback( async () => {
    try {
      const res = await getLabels();
      setLabel(res);
    } catch (err) {
      console.error(err);
      setMessage("Error");
    }
  },[]);

  const clear = async (text, reset) => {
    if (!ctxRef.current) return;
    if (INVERT) {
      ctxRef.current.fillStyle = "white";
      ctxRef.current.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    } else {
      ctxRef.current.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
    if (reset) await setRandomLabels();
    setMessage(text);
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

  const handleSubmit = async () => {
    try {
      setDisabled(true);
      setMessage("checking");

      let img = tf.browser.fromPixels(canvasRef.current, 1).toFloat().div(255.0);
      img = INVERT ? tf.sub(1.0, img) : img;

      const tensorData = {
        data: Array.from(new Uint8Array(img.mul(255).dataSync())),
        shape: img.shape
      };

      const result = await getClassify(tensorData);
      img.dispose();

      clear(result.correct ? "Correct" : "Incorrect", true);

    } catch (err) {
      console.error(err);
      setMessage("Error during prediction");
    } finally {
      setDisabled(false);
    }
  };

  return (

    <div className="hr-container d-flex flex-column" id="container">

      <h1 className="text-center mb-2">Handwritten recognition</h1>

      <div id="canvas-wrapper" className="mb-4 text-center">

        <canvas ref={canvasRef} className="border rounded shadow quad"/>

        <div className="fill"></div>

      </div>

      <div className="d-flex flex-wrap justify-content-center mb-4">

        <button className="btn btn-success m-2" onClick={() => clear("Draw the word", true)}>
          
          Reset

        </button>

        <button className="btn btn-success m-2" disabled={disabled} onClick={handleSubmit}>

          Submit

        </button>

      </div>

      <div className="output-container mb-3">

        <div className="label-grid">

          <Image width="250" height="100" src={label} unoptimized alt="spinner"/>

        </div>

      </div>

      <div className="text-center alert alert-success p-2 w-100 mb-0" role="alert">

        {message}

      </div>

    </div>

  );

};

export default Captcha;
