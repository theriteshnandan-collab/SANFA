'use client';

import { motion } from 'framer-motion';

export default function SovereignHero() {
  return (
    <div className="relative w-full max-w-2xl mx-auto aspect-[4/3] flex items-center justify-center overflow-hidden">
      <svg
        viewBox="0 0 800 600"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full text-black"
      >
        {/* The Lounge Chair (Minimalist Line) */}
        <motion.path
          d="M200,450 Q250,450 300,400 L450,250 Q500,200 450,150 L400,100"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2, ease: "easeInOut" }}
        />

        {/* The Character (Girl - Abstract Fine Line) */}
        <g className="opacity-80">
          {/* Body/Torso */}
          <motion.path
            d="M320,380 Q350,350 380,300 Q400,250 390,200"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2, delay: 0.5, ease: "easeInOut" }}
          />
          {/* Head */}
          <motion.circle
            cx="400"
            cy="160"
            r="25"
            stroke="currentColor"
            strokeWidth="1.5"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1.5 }}
          />
          {/* Arms holding phone */}
          <motion.path
            d="M360,280 Q380,300 410,290"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1.8 }}
          />
          {/* The Phone (Small Rectangle) */}
          <motion.rect
            x="415"
            y="270"
            width="15"
            height="30"
            rx="2"
            stroke="currentColor"
            strokeWidth="1"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 2.2 }}
          />
        </g>

        {/* The Hot Pink Shield Arc (Emotional Core) */}
        <motion.path
          d="M150,300 A250,250 0 0,1 650,300"
          stroke="#FF0066"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray="10 10"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ 
            pathLength: 1, 
            opacity: 0.6,
            strokeDashoffset: [0, -20]
          }}
          transition={{ 
            pathLength: { duration: 2.5, ease: "easeInOut" },
            opacity: { duration: 1, delay: 1 },
            strokeDashoffset: { duration: 2, repeat: Infinity, ease: "linear" }
          }}
        />

        {/* Floating security nodes */}
        <motion.circle
          cx="120"
          cy="280"
          r="4"
          fill="#FF0066"
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle
          cx="680"
          cy="320"
          r="4"
          fill="#FF0066"
          animate={{ y: [0, 15, 0] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
      </svg>
      
      {/* Ambient Glow */}
      <div className="absolute inset-0 bg-radial-gradient from-transparent to-white/10 pointer-events-none" />
    </div>
  );
}
