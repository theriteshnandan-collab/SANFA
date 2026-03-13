'use client';

import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect } from 'react';

export default function SovereignHero() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Smooth springs for cursor parallax
  const springX = useSpring(mouseX, { stiffness: 100, damping: 30 });
  const springY = useSpring(mouseY, { stiffness: 100, damping: 30 });

  // Parallax transformations
  const heroX = useTransform(springX, [-500, 500], [-10, 10]);
  const heroY = useTransform(springY, [-500, 500], [-10, 10]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      mouseX.set(clientX - innerWidth / 2);
      mouseY.set(clientY - innerHeight / 2);
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="relative w-full max-w-2xl mx-auto aspect-[4/3] flex items-center justify-center overflow-hidden cursor-default group">
      {/* Background Zen Pattern */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
           style={{ backgroundImage: 'radial-gradient(circle, #000 1px, transparent 1px)', backgroundSize: '30px 30px' }} />

      <motion.div
        className="relative z-10 w-full h-full flex items-center justify-center pointer-events-none"
        style={{ x: heroX, y: heroY }}
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {/* God-Level Hand-Drawn SVG Illustration (Pure Code) */}
        <svg viewBox="0 0 800 600" className="w-[90%] h-auto preserve-3d">
          <defs>
            <filter id="pencil" x="-20%" y="-20%" width="140%" height="140%">
              <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="3" result="noise" />
              <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.5" xChannelSelector="R" yChannelSelector="G" />
            </filter>
          </defs>

          {/* The Shield Arc (Geometric Protection) */}
          <motion.path
            d="M150,550 C150,550 50,300 150,50 C150,50 400,0 650,50 C650,50 750,300 650,550"
            fill="none"
            stroke="#FF0066"
            strokeWidth="0.5"
            strokeDasharray="5,10"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 3, ease: "easeInOut" }}
          />

          {/* Human Outline (Simplified Artisanal Style) */}
          <g filter="url(#pencil)" stroke="#111" fill="none" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
            {/* Chair Contour */}
            <path d="M250,500 L550,500 C580,500 600,480 600,450 L580,250 C570,220 540,200 510,210 L300,280 C270,290 250,310 250,340 Z" />
            
            {/* Person Relaxed (Flowing Artist Lines) */}
            <motion.g
              animate={{ 
                rotate: [-0.5, 0.5, -0.5],
                y: [0, -2, 0]
              }}
              transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
            >
                {/* Torso & Head */}
                <path d="M400,350 Q410,300 405,250 Q400,200 430,180 Q460,160 440,140 Q420,130 400,145 Q380,160 385,185 Q390,210 395,250" />
                {/* Hair (Lemonade Quirk) */}
                <path d="M430,140 Q450,120 440,100 Q430,90 415,105 Q400,120 405,135 M420,100 Q410,80 400,90" />
                {/* Legs (Relaxed) */}
                <path d="M400,350 Q450,370 500,380 Q550,390 580,450 M410,360 Q430,420 480,440" />
                {/* Arms & Phone */}
                <path d="M405,250 Q360,270 350,320 Q340,350 380,360" />
                <rect x="340" y="320" width="15" height="25" rx="2" transform="rotate(-15 347.5 332.5)" />
            </motion.g>
          </g>

          {/* Electronic Protection Nodes (Blinking S-Tier) */}
          {[
            { cx: 150, cy: 300 },
            { cx: 650, cy: 300 },
            { cx: 400, cy: 50 },
          ].map((node, i) => (
            <motion.circle
              key={i}
              cx={node.cx}
              cy={node.cy}
              r="4"
              fill="#FF0066"
              animate={{ opacity: [0.2, 1, 0.2], scale: [1, 1.5, 1] }}
              transition={{ duration: 3, repeat: Infinity, delay: i * 1 }}
            />
          ))}
        </svg>

        {/* Dynamic Glow Bloom */}
        <div className="absolute w-[80%] h-[80%] bg-[#FF0066]/5 rounded-full filter blur-[120px] pointer-events-none" />
      </motion.div>
      
      {/* Premium Finish Overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-white/10 to-transparent pointer-events-none" />
    </div>
  );
}
