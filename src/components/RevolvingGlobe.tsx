'use client';

import { motion, useAnimationFrame } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

// God-Level Constants for the Cyber-Sphere
const DOT_COUNT = 1000;
const RADIUS = 180;
const CANVAS_SIZE = 480;

interface Point {
  phi: number;
  theta: number;
  size: number;
  alpha: number;
}

export default function RevolvingGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState(0);
  const pointsRef = useRef<Point[]>([]);

  // Generate the spherical point cloud once
  useEffect(() => {
    const points: Point[] = [];
    for (let i = 0; i < DOT_COUNT; i++) {
      const phi = Math.acos(-1 + (2 * i) / DOT_COUNT);
      const theta = Math.sqrt(DOT_COUNT * Math.PI) * phi;
      points.push({
        phi,
        theta,
        size: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.5 + 0.2
      });
    }
    pointsRef.current = points;
  }, []);

  // Kinetic rotation engine
  useAnimationFrame((time) => {
    setRotation(time * 0.0004);
  });

  // 3D Projection & Rendering Engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Hi-DPI Scaling Protocol
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_SIZE * dpr;
    canvas.height = CANVAS_SIZE * dpr;
    ctx.scale(dpr, dpr);

    const centerX = CANVAS_SIZE / 2;
    const centerY = CANVAS_SIZE / 2;

    const render = () => {
      ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      
      // Sort points by Z-index for simplified depth transparency
      const sortedPoints = [...pointsRef.current].map(p => {
        const currentTheta = p.theta + rotation;
        const z = RADIUS * Math.sin(p.phi) * Math.sin(currentTheta);
        return { ...p, z, currentTheta };
      }).sort((a, b) => a.z - b.z);

      sortedPoints.forEach((p) => {
        const x = RADIUS * Math.sin(p.phi) * Math.cos(p.currentTheta);
        const y = RADIUS * Math.cos(p.phi);
        const z = p.z;

        // Perspective Projection Logic
        const fov = 500;
        const factor = fov / (fov + z);
        const x2d = x * factor + centerX;
        const y2d = y * factor + centerY;

        // Lighting & Visibility Protocol
        // z > 0 is front, z < 0 is back
        const baseOpacity = p.alpha * factor;
        const finalOpacity = z < 0 ? baseOpacity * 0.15 : baseOpacity;
        
        if (finalOpacity > 0.05) {
          ctx.beginPath();
          ctx.arc(x2d, y2d, p.size * factor, 0, Math.PI * 2);
          
          // Color Mapping: Lemonade Pink for data points
          ctx.fillStyle = z < 0 ? `rgba(0,0,0,0.05)` : `rgba(255, 0, 102, ${finalOpacity})`;
          ctx.fill();
        }
      });
    };

    render();
  }, [rotation]);

  return (
    <div className="relative w-full max-w-xl aspect-square flex items-center justify-center overflow-visible group scale-110">
      {/* 3D Mathematical Core (Pixel-Perfect Data Sphere) */}
      <canvas 
        ref={canvasRef}
        style={{ width: `${CANVAS_SIZE}px`, height: `${CANVAS_SIZE}px` }}
        className="relative z-10 filter drop-shadow-[0_25px_60px_rgba(255,0,102,0.15)]"
      />

      {/* Kinetic Orbital Shields (Restored & Enhanced Code SVGs) */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible z-20">
        <defs>
          <linearGradient id="cyber-orbit-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FF0066" stopOpacity="0.3" />
            <stop offset="50%" stopColor="#FF0066" stopOpacity="0" />
            <stop offset="100%" stopColor="#FF0066" stopOpacity="0.3" />
          </linearGradient>
        </defs>
        
        {[
          { rx: 240, ry: 90, rotate: 20, duration: 20 },
          { rx: 220, ry: 80, rotate: -40, duration: 30, reverse: true },
          { rx: 250, ry: 100, rotate: 70, duration: 45 },
        ].map((orbit, i) => (
          <motion.ellipse
            key={i}
            cx="50%"
            cy="50%"
            rx={orbit.rx}
            ry={orbit.ry}
            fill="none"
            stroke="url(#cyber-orbit-grad)"
            strokeWidth="0.8"
            strokeDasharray="4 8"
            style={{ transformOrigin: 'center', rotate: orbit.rotate }}
            animate={{ rotateZ: orbit.reverse ? -360 : 360 }}
            transition={{ duration: orbit.duration, repeat: Infinity, ease: "linear" }}
          />
        ))}
      </svg>

      {/* Atmospheric Bloom & Depth Shaders (Code-only) */}
      <div className="absolute w-[70%] h-[70%] bg-gradient-to-tr from-[#FF0066]/10 to-transparent rounded-full filter blur-[120px] pointer-events-none z-0 mix-blend-screen animate-pulse" />
    </div>
  );
}
