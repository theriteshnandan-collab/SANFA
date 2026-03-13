'use client';

import { motion } from 'framer-motion';

export default function SovereignHero() {
  return (
    <div className="relative w-full max-w-2xl mx-auto aspect-[4/3] flex items-center justify-center overflow-hidden">
      {/* Background Zen Pattern */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
           style={{ backgroundImage: 'radial-gradient(circle, #000 1px, transparent 1px)', backgroundSize: '30px 30px' }} />

      <motion.div
        className="relative z-10 w-full h-full flex items-center justify-center"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {/* The God-Tier Illustration */}
        <motion.img
          src="/assets/god-hero.png"
          alt="Secure and Relaxed with Sanfa"
          className="w-full h-full object-contain drop-shadow-[0_20px_50px_rgba(0,0,0,0.05)]"
          animate={{ 
            y: [0, -15, 0],
          }}
          transition={{ 
            duration: 8, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Dynamic Glimmer Dots (Semantic Security) */}
        {[
          { top: '20%', left: '15%', delay: 0 },
          { top: '80%', left: '85%', delay: 2 },
          { top: '10%', left: '80%', delay: 4 },
        ].map((node, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-[#FF0066] rounded-full blur-[1px]"
            style={{ top: node.top, left: node.left }}
            animate={{ 
              scale: [1, 1.5, 1],
              opacity: [0.3, 0.7, 0.3]
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity, 
              delay: node.delay,
              ease: "easeInOut" 
            }}
          />
        ))}

        {/* Ambient Ring */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[90%] h-[90%] border border-[#FF0066]/5 rounded-full pointer-events-none" />
      </motion.div>
      
      {/* Subtle Grain Overlay */}
      <div className="absolute inset-0 bg-white/[0.02] mix-blend-overlay pointer-events-none" />
    </div>
  );
}
