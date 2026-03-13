'use client';

import { motion } from 'framer-motion';

interface LineIllustrationProps {
  className?: string;
}

export default function LineIllustration({ className }: LineIllustrationProps) {
  return (
    <div className={`relative flex items-center justify-center overflow-hidden ${className}`}>
      <motion.div
        className="relative w-full h-full flex items-center justify-center p-8"
        initial={{ opacity: 0, pathLength: 0 }}
        animate={{ opacity: 1, pathLength: 1 }}
        transition={{ duration: 2, ease: "easeInOut" }}
      >
        {/* The God-Tier Abstract Art */}
        <motion.img
          src="/assets/god-feature.png"
          alt="Abstract Protection Wave"
          className="w-full h-auto object-contain mix-blend-multiply opacity-80"
          animate={{ 
            rotate: [0, 1, 0, -1, 0],
            scale: [1, 1.02, 1],
          }}
          transition={{ 
            duration: 10, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Semantic Nodes */}
        <motion.div
          className="absolute w-3 h-3 bg-[#FF0066] rounded-full blur-[2px]"
          style={{ top: '60%', right: '25%' }}
          animate={{ scale: [1, 1.2, 1], opacity: [0.4, 0.8, 0.4] }}
          transition={{ duration: 3, repeat: Infinity }}
        />
        
        {/* Ambient Ring */}
        <div className="absolute inset-0 border border-black/[0.03] rounded-3xl pointer-events-none" />
      </motion.div>
    </div>
  );
}
