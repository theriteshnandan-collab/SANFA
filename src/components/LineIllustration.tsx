'use client';

import { motion } from 'framer-motion';

interface LineIllustrationProps {
  className?: string;
}

export default function LineIllustration({ className }: LineIllustrationProps) {
  return (
    <div className={`relative flex items-center justify-center overflow-hidden rounded-[40px] bg-white border border-gray-50 ${className}`}>
      <motion.div
        className="relative w-full h-full flex items-center justify-center p-16"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {/* Layer 1: High-Fidelity Protection Wave (PNG) */}
        <motion.img
          src="/assets/god-feature.png"
          alt="Protection Protocol"
          className="w-full h-auto object-contain mix-blend-multiply opacity-80"
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

        {/* Layer 2: Geometric Security Grid (Standardized & Clean) */}
        <div className="absolute inset-8 border-[0.5px] border-black/[0.03] rounded-[30px] pointer-events-none" />
        <div className="absolute inset-16 border-[0.5px] border-black/[0.02] rounded-[20px] pointer-events-none" />

        {/* Dynamic Security Pulse (S-Tier Bloom) */}
        <motion.div
          className="absolute w-2.5 h-2.5 rounded-full pointer-events-none"
          style={{ 
            top: '55%', 
            right: '25%',
            backgroundColor: '#FF0066',
            boxShadow: '0 0 20px rgba(255, 0, 102, 0.6)'
          }}
          animate={{ 
            scale: [1, 1.8, 1], 
            opacity: [0.4, 1, 0.4],
          }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* Secure Label (Sanfa Shito) */}
        <div className="absolute bottom-8 left-8">
           <span className="text-[10px] font-black uppercase tracking-[0.4em] text-gray-300">PROTOCOL: AES-512</span>
        </div>
      </motion.div>
    </div>
  );
}
