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
        {/* The God-Tier Abstract Art */}
        <motion.img
          src="/assets/god-feature.png"
          alt="Abstract Protection Wave"
          className="w-full h-auto object-contain mix-blend-multiply opacity-80 filter drop-shadow-[0_10px_30px_rgba(0,0,0,0.03)]"
          animate={{ 
            rotate: [0, 0.5, 0, -0.5, 0],
            scale: [1, 1.01, 1],
          }}
          transition={{ 
            duration: 12, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Semantic Node (Bloom) */}
        <motion.div
          className="absolute w-2 h-2 rounded-full pointer-events-none"
          style={{ 
            top: '62%', 
            right: '28%',
            backgroundColor: '#FF0066',
            boxShadow: '0 0 10px rgba(255, 0, 102, 0.4)'
          }}
          animate={{ 
            scale: [1, 1.3, 1], 
            opacity: [0.3, 0.8, 0.3],
            filter: ['blur(0px)', 'blur(1.5px)', 'blur(0px)']
          }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Border Detail */}
        <div className="absolute inset-4 border border-black/[0.02] rounded-2xl pointer-events-none" />
      </motion.div>
    </div>
  );
}
