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
        initial={{ opacity: 0, scale: 0.98 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {/* The God-Tier High-Fidelity Artist Wave (Restored PNG) */}
        <motion.img
          src="/assets/god-feature.png"
          alt="Abstract Protection Wave"
          className="w-full h-auto object-contain mix-blend-multiply opacity-80 filter drop-shadow-[0_15px_40px_rgba(0,0,0,0.04)]"
          animate={{ 
            rotate: [0, 0.8, 0, -0.8, 0],
            scale: [1, 1.02, 1],
          }}
          transition={{ 
            duration: 15, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Semantic Node (Electric Pulse) */}
        <motion.div
          className="absolute w-2.5 h-2.5 rounded-full pointer-events-none"
          style={{ 
            bottom: '35%', 
            right: '25%',
            backgroundColor: '#FF0066',
            boxShadow: '0 0 15px rgba(255, 0, 102, 0.5)'
          }}
          animate={{ 
            scale: [1, 1.5, 1], 
            opacity: [0.4, 1, 0.4],
            filter: ['blur(0.5px)', 'blur(2px)', 'blur(0.5px)']
          }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Subtle Depth Detail */}
        <div className="absolute inset-6 border border-black/[0.02] rounded-3xl pointer-events-none" />
      </motion.div>
    </div>
  );
}
