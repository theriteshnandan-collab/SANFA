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
        {/* Subtle Background Pattern (Zen dots) */}
        <pattern id="pattern-dots" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
          <circle cx="2" cy="2" r="1" fill="currentColor" fillOpacity="0.05" />
        </pattern>
        <rect width="800" height="600" fill="url(#pattern-dots)" />

        {/* The Lounge Chair (Intricate Minimalist Line) */}
        <motion.path
          d="M180,480 Q250,480 320,420 L480,260 Q530,210 480,160 L420,110 C400,90 380,100 370,120"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2.5, ease: "easeInOut" }}
        />
        <motion.path
          d="M200,485 L180,510 M500,240 L520,265"
          stroke="currentColor"
          strokeWidth="0.8"
          strokeLinecap="round"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.3 }}
          transition={{ delay: 2 }}
        />

        {/* The Character (Girl - Detailed Fine Line) */}
        <g className="character-group">
          {/* Detailed Hair (Whimsical curls) */}
          <motion.path
            d="M440,140 C460,120 480,140 470,170 C480,180 470,210 450,200"
            stroke="currentColor"
            strokeWidth="1"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1.5, delay: 1.5 }}
          />
          
          {/* Face Outline & Peaceful Expression */}
          <motion.path
            d="M435,165 C440,195 420,210 400,205"
            stroke="currentColor"
            strokeWidth="1.2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1 }}
          />
          <motion.path
            d="M410,185 Q415,190 420,185"
            stroke="currentColor"
            strokeWidth="0.8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2.2 }}
          />

          {/* Torso & Relaxed Posture */}
          <motion.path
            d="M400,205 Q350,250 330,350 Q320,400 350,430 L400,450"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2, delay: 0.5 }}
          />

          {/* Arms holding phone (Articulated) */}
          <motion.path
            d="M360,280 C360,320 380,310 420,300"
            stroke="currentColor"
            strokeWidth="1.2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1.8 }}
          />
          
          {/* Hand holding phone detail */}
          <motion.path
            d="M415,295 C410,290 410,280 420,275"
            stroke="currentColor"
            strokeWidth="1"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2.5 }}
          />

          {/* The Phone (Sleek minimalist) */}
          <motion.rect
            x="425"
            y="270"
            width="18"
            height="35"
            rx="3"
            stroke="currentColor"
            strokeWidth="1"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 2.7 }}
          />
          <motion.path
            d="M432,275 L436,275"
            stroke="currentColor"
            strokeWidth="0.5"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.4 }}
            transition={{ delay: 3 }}
          />
        </g>

        {/* The Hot Pink Shield Arc (Master-Craft Fidelity) */}
        <motion.path
          d="M120,320 C120,120 680,120 680,320"
          stroke="#FF0066"
          strokeWidth="2.5"
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 0.8 }}
          transition={{ duration: 3, ease: "circOut", delay: 1 }}
        />
        <motion.path
          d="M150,320 C150,180 650,180 650,320"
          stroke="#FF0066"
          strokeWidth="1"
          strokeLinecap="round"
          strokeDasharray="4 8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.4 }}
          transition={{ duration: 2, delay: 2 }}
        />

        {/* Floating Semantic Nodes (Privacy, Encryption, Speed) */}
        {[
          { cx: 100, cy: 300, delay: 0 },
          { cx: 700, cy: 350, delay: 1 },
          { cx: 400, cy: 80, delay: 2 }
        ].map((node, i) => (
          <motion.g key={i} animate={{ y: [0, -10, 0] }} transition={{ duration: 4, repeat: Infinity, delay: node.delay }}>
            <circle cx={node.cx} cy={node.cy} r="3" fill="#FF0066" />
            <circle cx={node.cx} cy={node.cy} r="10" stroke="#FF0066" strokeWidth="0.5" strokeOpacity="0.2" />
          </motion.g>
        ))}
      </svg>
      
      {/* Texture grain overlay */}
      <div className="absolute inset-0 bg-[#000000]/[0.02] mix-blend-overlay pointer-events-none" />
    </div>
  );
}
