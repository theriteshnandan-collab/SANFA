'use client';

import { motion, useAnimationFrame } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

// God-Level Constants for the Cyber-Sphere
const DOT_COUNT = 1200;
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

  // Generate the spherical point cloud with high density
  useEffect(() => {
    const points: Point[] = [];
    for (let i = 0; i < DOT_COUNT; i++) {
      const phi = Math.acos(-1 + (2 * i) / DOT_COUNT);
      const theta = Math.sqrt(DOT_COUNT * Math.PI) * phi;
      points.push({
        phi,
        theta,
        size: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.4 + 0.15
      });
    }
    pointsRef.current = points;
  }, []);

  // Kinetic rotation engine (Smooth & Persistent)
  useAnimationFrame((time) => {
    setRotation(time * 0.0003);
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
      
      // Calculate depth and projection for all points
      const projected = pointsRef.current.map(p => {
        const currentTheta = p.theta + rotation;
        const x = RADIUS * Math.sin(p.phi) * Math.cos(currentTheta);
        const y = RADIUS * Math.cos(p.phi);
        const z = RADIUS * Math.sin(p.phi) * Math.sin(currentTheta);

        const fov = 500;
        const factor = fov / (fov + z);
        const x2d = x * factor + centerX;
        const y2d = y * factor + centerY;

        return { x2d, y2d, factor, z, size: p.size, alpha: p.alpha };
      }).sort((a, b) => a.z - b.z); // Z-sorting for depth

      projected.forEach((p) => {
        // Opacity Mapping: S-Tier Depth Perception
        let opacity = p.alpha * p.factor;
        if (p.z < 0) opacity *= 0.1; // Ghost back points
        
        if (opacity > 0.02) {
          ctx.beginPath();
          ctx.arc(p.x2d, p.y2d, p.size * p.factor, 0, Math.PI * 2);
          
          // Color Mapping: Pure Cyber Pink
          ctx.fillStyle = p.z < 0 ? `rgba(0,0,0,0.03)` : `rgba(255, 0, 102, ${opacity})`;
          ctx.fill();
        }
      });
    };

    render();
  }, [rotation]);

  return (
    <div className="relative w-full max-w-xl aspect-square flex items-center justify-center overflow-visible group scale-110">
      {/* 3D Cyber-Sphere Core (Pure Code) */}
      <canvas 
        ref={canvasRef}
        style={{ width: `${CANVAS_SIZE}px`, height: `${CANVAS_SIZE}px` }}
        className="relative z-10 transition-transform duration-1000 group-hover:scale-105"
      />

      {/* Atmospheric Shaders (Geometric & Clean) */}
      <div className="absolute w-[80%] h-[80%] bg-gradient-to-tr from-[#FF0066]/5 to-transparent rounded-full filter blur-[120px] pointer-events-none z-0" />
      
      {/* Laser Protocol (Minimalist Sentiment) */}
      <motion.div 
        className="absolute w-[105%] h-[105%] border-[0.5px] border-[#FF0066]/10 rounded-full z-0"
        animate={{ rotateZ: 360 }}
        transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}
