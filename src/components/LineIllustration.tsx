'use client';

import { motion } from 'framer-motion';

interface LineIllustrationProps {
  className?: string;
}

export default function LineIllustration({ className }: LineIllustrationProps) {
  return (
    <div className={`relative flex items-center justify-center overflow-hidden rounded-3xl ${className}`}>
      <motion.div
        className="relative w-full h-full flex items-center justify-center p-12"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {/* God-Tier Abstract Artistic Wave (Pure Code SVG) */}
        <svg viewBox="0 0 400 400" className="w-full h-auto opacity-70">
          <motion.path
            d="M50,200 Q100,100 150,200 T250,200 T350,200"
            fill="none"
            stroke="#111"
            strokeWidth="0.8"
            initial={{ pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            transition={{ duration: 2, ease: "easeInOut" }}
          />
          <motion.path
            d="M50,220 Q120,120 180,220 T300,220 T350,220"
            fill="none"
            stroke="#FF0066"
            strokeWidth="0.4"
            opacity="0.5"
            initial={{ pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            transition={{ duration: 2.5, ease: "easeInOut", delay: 0.2 }}
          />
          <motion.path
            d="M30,180 Q150,250 250,150 T380,180"
            fill="none"
            stroke="#111"
            strokeWidth="0.5"
            strokeDasharray="4 4"
            initial={{ pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            transition={{ duration: 3, ease: "easeInOut", delay: 0.4 }}
          />
          
          {/* Floating Security Particles */}
          {[
            { cx: 100, cy: 150, delay: 0 },
            { cx: 300, cy: 250, delay: 1 },
            { cx: 200, cy: 100, delay: 2 },
          ].map((dot, i) => (
            <motion.circle
              key={i}
              cx={dot.cx}
              cy={dot.cy}
              r="2"
              fill="#FF0066"
              animate={{ 
                y: [0, -10, 0],
                opacity: [0.3, 0.8, 0.3] 
              }}
              transition={{ duration: 4, repeat: Infinity, delay: dot.delay }}
            />
          ))}
        </svg>

        {/* Semantic Bloom */}
        <motion.div
          className="absolute w-2 h-2 rounded-full pointer-events-none"
          style={{ 
            top: '60%', 
            right: '30%',
            backgroundColor: '#FF0066',
            boxShadow: '0 0 10px rgba(255, 0, 102, 0.4)'
          }}
          animate={{ 
            scale: [1, 1.4, 1], 
            opacity: [0.4, 1, 0.4]
          }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Border Detail */}
        <div className="absolute inset-4 border border-black/[0.03] rounded-2xl pointer-events-none" />
      </motion.div>
    </div>
  );
}
