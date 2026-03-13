'use client';

import { motion } from 'framer-motion';

export default function RevolvingGlobe() {
  return (
    <div className="relative w-full max-w-lg aspect-square flex items-center justify-center overflow-hidden group">
      {/* The Sphere Shell */}
      <div className="w-[85%] h-[85%] rounded-full relative overflow-hidden bg-white shadow-[0_30px_100px_-20px_rgba(0,0,0,0.1)] transition-all duration-700 group-hover:scale-[1.05] group-hover:shadow-[0_40px_150px_-20px_rgba(255,0,102,0.15)]">
        
        {/* Depth & Shine Overlays (Premium Glassmorphism) */}
        <div className="absolute inset-0 z-20 rounded-full shadow-[inset_-40px_-40px_100px_rgba(0,0,0,0.05),inset_40px_40px_100px_white]" />
        <div className="absolute inset-0 z-30 rounded-full border-[0.5px] border-black/5 pointer-events-none" />
        
        {/* God-Tier Map Revolution (Seamless Protocol) */}
        <div className="w-full h-full flex absolute top-0 left-0">
          <motion.div 
            className="flex h-full w-[200%] transform-gpu"
            animate={{ x: ["0%", "-50%"] }}
            transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
          >
            {/* Detailed Artist Map Segment 1 */}
            <div className="w-1/2 h-full opacity-[0.9]">
              <img 
                src="/assets/god-globe.png" 
                alt="World Map Segment 1" 
                className="w-full h-full object-cover scale-150 transform-gpu" 
              />
            </div>
            {/* Seamless Loop Segment 2 */}
            <div className="w-1/2 h-full opacity-[0.9]">
              <img 
                src="/assets/god-globe.png" 
                alt="World Map Segment 2" 
                className="w-full h-full object-cover scale-150 transform-gpu" 
              />
            </div>
          </motion.div>
        </div>

        {/* Atmospheric Glow & Shine */}
        <div className="absolute -top-[10%] -left-[10%] w-[60%] h-[60%] bg-white/40 filter blur-3xl pointer-events-none z-40" />
        <div className="absolute bottom-[10%] right-[10%] w-32 h-32 bg-[#FF0066]/5 filter blur-3xl pointer-events-none z-40" />
      </div>

      {/* Kinetic Orbital Protocols (Artist weights) */}
      {[
        { rotateX: 75, rotateY: 15, duration: 25, size: '100%', color: '#FF0066' },
        { rotateX: -60, rotateY: 30, duration: 40, size: '92%', reverse: true, color: '#FF0066' },
        { rotateX: 25, rotateY: -75, duration: 60, size: '115%', color: '#111111' },
      ].map((orbit, i) => (
        <motion.div 
          key={i}
          className="absolute border-[0.6px] rounded-full pointer-events-none"
          style={{ 
            width: orbit.size, 
            height: orbit.size,
            rotateX: orbit.rotateX, 
            rotateY: orbit.rotateY,
            borderColor: orbit.color,
            opacity: 0.15
          }}
          animate={{ rotateZ: orbit.reverse ? -360 : 360 }}
          transition={{ duration: orbit.duration, repeat: Infinity, ease: "linear" }}
        />
      ))}

      {/* Floating Spark Node */}
      <motion.div
        className="absolute w-2 h-2 bg-[#FF0066] rounded-full filter blur-[0.5px] shadow-[0_0_8px_#FF0066]"
        animate={{ 
          rotateY: 360,
          scale: [1, 1.4, 1],
          opacity: [0.6, 1, 0.6]
        }}
        transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
        style={{ transformOrigin: '220px center' }}
      />
    </div>
  );
}
