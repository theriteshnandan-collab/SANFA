'use client';

import { motion, useAnimationFrame, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

// God-Level Mathematical constants for the sphere
const DOT_COUNT = 800;
const RADIUS = 160;

export default function RevolvingGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState(0);
  
  // Kinetic setup
  useAnimationFrame((time) => {
    setRotation(time * 0.0005);
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set high DPI scale
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 400 * dpr;
    canvas.height = 400 * dpr;
    ctx.scale(dpr, dpr);

    // Generate static point cloud (Lat/Lon)
    // We only do this once
    const points: { phi: number; theta: number; size: number }[] = [];
    for (let i = 0; i < DOT_COUNT; i++) {
        // Uniform distribution on sphere
        const phi = Math.acos(-1 + (2 * i) / DOT_COUNT);
        const theta = Math.sqrt(DOT_COUNT * Math.PI) * phi;
        
        // Slightly vary point size for "depth"
        const size = Math.random() * 1.5 + 0.5;
        points.push({ phi, theta, size });
    }

    const render = () => {
        ctx.clearRect(0, 0, 400, 400);
        
        const centerX = 200;
        const centerY = 200;

        points.forEach((p) => {
            // Add rotation to theta
            const currentTheta = p.theta + rotation;
            
            // 3D Cartesian coordinates
            const x = RADIUS * Math.sin(p.phi) * Math.cos(currentTheta);
            const y = RADIUS * Math.cos(p.phi);
            const z = RADIUS * Math.sin(p.phi) * Math.sin(currentTheta);

            // Simple perspective projection
            const fov = 400;
            const factor = fov / (fov + z);
            const x2d = x * factor + centerX;
            const y2d = y * factor + centerY;

            // Only draw points on the front hemisphere for a "solid" look
            // but for a "Cyber/Protective" look, we can draw back points with low opacity
            const opacity = z < 0 ? 0.05 : factor;
            
            if (opacity > 0.1) {
                ctx.beginPath();
                ctx.arc(x2d, y2d, p.size * factor, 0, Math.PI * 2);
                
                // Content-aware coloring (Pure Lemonade Pink for front, Grey for ghosting)
                ctx.fillStyle = z < 0 ? 'rgba(0,0,0,0.1)' : `rgba(255, 0, 102, ${opacity})`;
                ctx.fill();
            }
        });

        // Loop is handled by React state update, but for pure performance we could use a local loop
        // However, React integration is better for component lifecycle here.
    };

    render();
  }, [rotation]);

  return (
    <div className="relative w-full max-w-lg aspect-square flex items-center justify-center overflow-hidden group">
      {/* 3D Canvas Engine */}
      <canvas 
        ref={canvasRef}
        style={{ width: '400px', height: '400px' }}
        className="relative z-10 drop-shadow-[0_20px_50px_rgba(255,0,102,0.1)]"
      />

      {/* Protective Orbital Protocol (SVG - Light & Kinetic) */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
        <defs>
          <linearGradient id="orbit-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FF0066" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#FF0066" stopOpacity="0" />
          </linearGradient>
        </defs>
        
        {/* Kinetic Rings */}
        {[
          { rx: 190, ry: 70, rotate: 15, duration: 15 },
          { rx: 180, ry: 60, rotate: -30, duration: 25, reverse: true },
          { rx: 200, ry: 80, rotate: 75, duration: 40 },
        ].map((orbit, i) => (
          <motion.ellipse
            key={i}
            cx="50%"
            cy="50%"
            rx={orbit.rx}
            ry={orbit.ry}
            fill="none"
            stroke="url(#orbit-grad)"
            strokeWidth="1"
            style={{ transformOrigin: 'center', rotate: orbit.rotate }}
            animate={{ rotateZ: orbit.reverse ? -360 : 360 }}
            transition={{ duration: orbit.duration, repeat: Infinity, ease: "linear" }}
          />
        ))}
      </svg>

      {/* Atmospheric Bloom */}
      <div className="absolute w-[60%] h-[60%] bg-[#FF0066]/5 rounded-full filter blur-[100px] pointer-events-none" />
    </div>
  );
}
