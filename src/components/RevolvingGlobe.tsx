'use client';

import { motion } from 'framer-motion';

export default function RevolvingGlobe() {
  return (
    <div className="relative w-full max-w-lg aspect-square flex items-center justify-center overflow-hidden group scale-110">
      {/* The Sphere Shell (Electronic Core) */}
      <div className="w-[88%] h-[88%] rounded-full relative overflow-hidden bg-white shadow-[0_45px_120px_-30px_rgba(0,0,0,0.15)] transition-all duration-1000 group-hover:scale-[1.04] group-hover:shadow-[0_60px_150px_-30px_rgba(255,0,102,0.2)]">
        
        {/* Superior Depth & Shine (Glassmorphism + OLED true-black shadows) */}
        <div className="absolute inset-0 z-30 rounded-full shadow-[inset_-60px_-60px_120px_rgba(0,0,0,0.08),inset_60px_60px_120px_white]" />
        <div className="absolute inset-0 z-40 rounded-full border-[0.5px] border-black/10 pointer-events-none" />
        
        {/* High-Fidelity Map Revolution (Triple-Segment Seamless Loop) */}
        <div className="w-full h-full flex absolute top-0 left-0">
          <motion.div 
            className="flex h-full w-[300%] transform-gpu"
            animate={{ x: ["0%", "-66.666%"] }}
            transition={{ duration: 45, repeat: Infinity, ease: "linear" }}
          >
            {/* Detailed Artist Map Segments */}
            {[1, 2, 3].map((seg) => (
              <div key={seg} className="w-1/3 h-full px-4 transform-gpu">
                <img 
                  src="/assets/god-globe.png" 
                  alt={`World Map Segment ${seg}`} 
                  className="w-full h-full object-contain filter brightness-[1.02] contrast-[1.05]" 
                />
              </div>
            ))}
          </motion.div>
        </div>

        {/* Dynamic Atmospheric Bloom & Electronic Glitch Effects */}
        <div className="absolute -top-[10%] -left-[10%] w-[70%] h-[70%] bg-white/50 filter blur-[80px] pointer-events-none z-50 mix-blend-soft-light" />
        <div className="absolute bottom-[5%] right-[5%] w-48 h-48 bg-[#FF0066]/10 filter blur-[100px] pointer-events-none z-50 animate-pulse" />
      </div>

      {/* Kinetic Orbital Protocols (Hyper-Dynamic Artist Weights) */}
      {[
        { rx: '100%', ry: '30%', rotateX: 75, rotateY: 15, duration: 25, color: '#FF0066', opacity: 0.2 },
        { rx: '92%', ry: '25%', rotateX: -60, rotateY: 30, duration: 35, reverse: true, color: '#FF0066', opacity: 0.15 },
        { rx: '110%', ry: '35%', rotateX: 25, rotateY: -75, duration: 55, color: '#111111', opacity: 0.1 },
      ].map((orbit, i) => (
        <motion.div 
          key={i}
          className="absolute border-[1px] rounded-full pointer-events-none"
          style={{ 
            width: orbit.rx, 
            height: orbit.ry,
            rotateX: orbit.rotateX, 
            rotateY: orbit.rotateY,
            borderColor: orbit.color,
            opacity: orbit.opacity
          }}
          animate={{ rotateZ: orbit.reverse ? -360 : 360 }}
          transition={{ duration: orbit.duration, repeat: Infinity, ease: "linear" }}
        />
      ))}

      {/* Floating Laser Node (Protection Sentiment) */}
      <motion.div
        className="absolute w-2.5 h-2.5 bg-[#FF0066] rounded-full filter blur-[0.8px] shadow-[0_0_15px_#FF0066] z-40"
        animate={{ 
          rotateY: 360,
          scale: [1, 1.3, 1],
          opacity: [0.7, 1, 0.7]
        }}
        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        style={{ transformOrigin: '240px center' }}
      />
    </div>
  );
}
